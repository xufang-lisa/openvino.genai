// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <thread>

#include "openvino/genai/text_streamer.hpp"
#include "speculative_decoding_impl.hpp"
#include "paged_attention_transformations.hpp"
#include "utils.hpp"


namespace ov::genai {
template<class... Ts> struct overloaded : Ts... {using Ts::operator()...;};
template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

bool are_tokenizers_equal(Tokenizer& lhs, Tokenizer& rhs) {
    std::string test_string = "Could you please tell me something about OpenVINO.GenAI?";
    ov::Tensor encoded_string_lhs = lhs.encode(test_string).input_ids,
               encoded_string_rhs = rhs.encode(test_string).input_ids;
    
    ov::Shape shape_lhs = encoded_string_lhs.get_shape(),
              shape_rhs = encoded_string_rhs.get_shape();

    return shape_lhs == shape_rhs && lhs.get_eos_token_id() == rhs.get_eos_token_id() &&
           lhs.get_bos_token_id() == rhs.get_bos_token_id() && lhs.get_pad_token_id() == rhs.get_pad_token_id();
}

ContinuousBatchingPipeline::SpeculativeDecodingImpl::SpeculativeDecodingImpl(const ov::genai::ModelDesc& main_model_desc, 
                                                                             const ov::genai::ModelDesc& draft_model_desc) {
    auto main_model = main_model_desc.model;
    auto draft_model = draft_model_desc.model;

    auto main_scheduler_config = main_model_desc.scheduler_config;
    auto main_device = main_model_desc.device;

    auto main_kv_cache_config = utils::apply_paged_attention_transformations(main_model, main_model_desc.scheduler_config.use_cache_eviction);
    auto draft_kv_cache_config = utils::apply_paged_attention_transformations(draft_model, main_model_desc.scheduler_config.use_cache_eviction);

    utils::apply_gather_before_matmul_transformation(main_model);
    utils::apply_gather_before_matmul_transformation(draft_model);

    std::string draft_device = draft_model_desc.device.empty() ? main_model_desc.device : draft_model_desc.device;
    bool is_draft_scheduler_undefined = draft_model_desc.scheduler_config == SchedulerConfig();

    ov::genai::SchedulerConfig main_scheduler_config_updated = main_scheduler_config,
                               draft_scheduler_config = is_draft_scheduler_undefined ? main_scheduler_config : draft_model_desc.scheduler_config;

    if (is_draft_scheduler_undefined) {
        // split KV cache to 2 caches for main and draft models
        auto compute_total_hidden_size = [] (const std::vector<KVHeadConfig>& kv_cache_config) -> size_t {
            size_t total_hidden_size = 0;
            for (auto & config : kv_cache_config) {
                total_hidden_size += config.k_head_size * config.num_k_heads + config.v_head_size * config.num_v_heads;
            }
            return total_hidden_size;
        };
        float main_model_hidden_size = compute_total_hidden_size(main_kv_cache_config),
              draft_model_hidden_size = compute_total_hidden_size(draft_kv_cache_config);
        auto k = draft_model_hidden_size / (main_model_hidden_size + draft_model_hidden_size);

        // TODO: work with KV blocks as it will be more precise instead of GBs
        size_t main_cache_size = std::ceil(main_scheduler_config.cache_size * (1.f - k)),
               draft_cache_size = main_scheduler_config.cache_size - main_cache_size;
        if (draft_cache_size == 0 && main_cache_size > 0) {
            main_cache_size -= (main_cache_size > 1 ? 1 : 0);
            draft_cache_size = 1;
        }

        main_scheduler_config_updated.cache_size = main_cache_size;
        draft_scheduler_config.cache_size = draft_cache_size;
    }

    ov::AnyMap draft_properties = draft_model_desc.properties.empty() ? main_model_desc.properties : draft_model_desc.properties;

    // main and draft model can have different tokenizers
    // to do: support retokenization: 154103
    Tokenizer main_model_tokenizer = main_model_desc.tokenizer;
    Tokenizer draft_model_tokenizer = draft_model_desc.tokenizer;

    // todo: remove this condition after support of CVS-154103
    OPENVINO_ASSERT(are_tokenizers_equal(main_model_tokenizer, draft_model_tokenizer), "Tokenizers for draft and main models are different!");
    
    m_tokenizer = main_model_tokenizer;

    // to create `main_pipeline` with enabled validation_mode and `draft_pipeline` with disabled validation mode
    m_main_pipeline = std::make_shared<ContinuousBatchingForSpeculativeDecodingImpl>(
        main_model, main_model_tokenizer, main_model_desc.generation_config,
        main_kv_cache_config, main_scheduler_config_updated, main_device, main_model_desc.properties, true);
    m_draft_pipeline = std::make_shared<ContinuousBatchingForSpeculativeDecodingImpl>(
        draft_model, draft_model_tokenizer, draft_model_desc.generation_config,
        draft_kv_cache_config, draft_scheduler_config, draft_device, draft_properties, false);

    m_perf_metrics = PerfMetrics();
    m_perf_metrics.raw_metrics.m_inference_durations =  {{ MicroSeconds(0.0f) }};

}

GenerationHandle
ContinuousBatchingPipeline::SpeculativeDecodingImpl::add_request(uint64_t request_id,
                                                                 const ov::Tensor& input_ids,
                                                                 ov::genai::GenerationConfig sampling_params) {
    m_sd_metrics.set_generated_len(request_id, sampling_params.get_max_new_tokens(input_ids.get_size()));
    std::lock_guard<std::mutex> lock(m_draft_generations_mutex);
    auto draft_sampling_params = sampling_params;
    draft_sampling_params.ignore_eos = true;
    draft_sampling_params.stop_strings = {};
    m_draft_generations.insert({request_id, m_draft_pipeline->add_request(request_id, input_ids, draft_sampling_params)});
    return m_main_pipeline->add_request(request_id, input_ids, sampling_params);
};

GenerationHandle
ContinuousBatchingPipeline::SpeculativeDecodingImpl::add_request(uint64_t request_id,
                                                                 const std::string& prompt,
                                                                 ov::genai::GenerationConfig sampling_params) {
    m_sd_metrics.set_generated_len(request_id, sampling_params.get_max_new_tokens(prompt.length()));
    std::lock_guard<std::mutex> lock(m_draft_generations_mutex);
    auto draft_sampling_params = sampling_params;
    draft_sampling_params.ignore_eos = true;
    draft_sampling_params.stop_strings = {};
    m_draft_generations.insert({request_id, m_draft_pipeline->add_request(request_id, prompt, draft_sampling_params)});
    return m_main_pipeline->add_request(request_id, prompt, sampling_params);
}

bool ContinuousBatchingPipeline::SpeculativeDecodingImpl::has_non_finished_requests() {
    return m_main_pipeline->has_non_finished_requests();
}

void print_generated_request(const ov::genai::GeneratedRequests& requests) {
    for (const auto& request : requests) {
        for (const auto& sequence : request.second) {
            std::cout << "request_id: " << request.first << " | sequence_id: " << sequence.first << " | ";
            for (const auto& token_id : sequence.second.token_ids) {
                std::cout << token_id << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}

void ContinuousBatchingPipeline::SpeculativeDecodingImpl::step() {
    // this blocks adding new requests during step as it may break coherence between main and draft models
    std::lock_guard<std::mutex> lock{m_draft_generations_mutex};

    auto& raw_perf_counters = m_perf_metrics.raw_metrics;

    ManualTimer step_timer("speculative_decoding: step()");
    step_timer.start();

    m_draft_pipeline->pull_awaiting_requests(true);
    m_main_pipeline->pull_awaiting_requests();

    // generate candidates by draft model
    ManualTimer draft_timer("speculative_decoding: draft_model: multistep()");
    draft_timer.start();
    m_draft_pipeline->multistep();
    draft_timer.end();
    m_sd_metrics.draft_duration += draft_timer.get_duration();
    m_pipeline_metrics = m_main_pipeline->get_metrics();

    // to generate num_matches statistic
    std::map<int64_t, UpdateRequestResult> update_sequence_info;
    // put candidates to model KV cache
    auto draft_generated_requests = m_draft_pipeline->get_generated_requests();
    for (const auto& candidate : m_draft_pipeline->get_generated_requests()) {
        auto update_result = m_main_pipeline->update_request(candidate.first, candidate.second, false);
        update_sequence_info.insert({{candidate.first, update_result}});
    }

    ManualTimer main_timer("speculative_decoding: main_model: step()");
    main_timer.start();
    m_main_pipeline->step();
    main_timer.end();
    m_sd_metrics.main_duration += main_timer.get_duration();
    m_pipeline_metrics = m_main_pipeline->get_metrics();

    auto main_generated_requests = m_main_pipeline->get_generated_requests();
    for (const auto& checked_sequence : main_generated_requests) {
        auto update_result = m_draft_pipeline->update_request(checked_sequence.first, checked_sequence.second, true);
        update_sequence_info[checked_sequence.first].removed_tokens_cnt = update_result.removed_tokens_cnt;
    }

    // finish draft request if the generation was completed
    for (const auto& draft_request : draft_generated_requests) {
        auto request_id = draft_request.first;
        if (!main_generated_requests.count(request_id)) {
            m_draft_pipeline->finish_request(request_id);
            // remove draft_generation_handle from queue
            m_draft_generations.erase(request_id);
        }
        auto updated_seq_info = update_sequence_info[request_id];
        // several prompt phase
        if (updated_seq_info.inserted_tokens_cnt == 0) {
            continue;
        }
        float acceptance_rate = 1 - static_cast<float>(updated_seq_info.removed_tokens_cnt) / updated_seq_info.inserted_tokens_cnt;
        m_sd_metrics.update_acceptance_rate(request_id, acceptance_rate * 100);
        m_sd_metrics.update_draft_accepted_tokens(request_id, (updated_seq_info.inserted_tokens_cnt - updated_seq_info.removed_tokens_cnt));
    }

    // update perf metrics
    const auto num_generated_tokens = m_main_pipeline->get_processed_tokens_per_iteration();
    if (num_generated_tokens > 0) {
        auto infer_duration = step_timer.get_duration_microsec();
    
        raw_perf_counters.m_token_infer_durations.emplace_back(infer_duration);
        raw_perf_counters.m_inference_durations[0] += MicroSeconds(infer_duration);
        raw_perf_counters.m_new_token_times.emplace_back(main_timer.get_end_time());

        raw_perf_counters.m_batch_sizes.emplace_back(num_generated_tokens);
    }

    if (main_generated_requests.empty() && 1) {
        m_draft_pipeline->get_infer_duration(m_sd_metrics.draft_infer_duration, m_sd_metrics.draft_infer_num);
        m_main_pipeline->get_infer_duration(m_sd_metrics.main_infer_duration, m_sd_metrics.main_infer_num);
        m_sd_metrics.print(true);
        m_sd_metrics.clean_up();
    }
    step_timer.end();
}

std::vector<EncodedGenerationResult>
ContinuousBatchingPipeline::SpeculativeDecodingImpl::generate(const std::vector<ov::Tensor>& input_ids,
                                                              const std::vector<GenerationConfig>& sampling_params,
                                                              const StreamerVariant& streamer) {
    m_perf_metrics = PerfMetrics();
    m_perf_metrics.raw_metrics.m_inference_durations =  {{ MicroSeconds(0.0f) }};
    m_sd_metrics.clean_up();
    OPENVINO_ASSERT(!has_non_finished_requests(), "Generate cannot be called while ContinuousBatchingPipeline is already in running state. Use ContinuousBatchingPipeline::add_request");
    OPENVINO_ASSERT(input_ids.size() == sampling_params.size());

    ManualTimer generate_timer("speculative_decoding: generate()");
    generate_timer.start();

    // checks that all requests has the same LoRA adapters property value
    for (size_t i = 1; i < sampling_params.size(); ++i) {
        OPENVINO_ASSERT(sampling_params[i - 1].adapters == sampling_params[i].adapters,
            "LoRA adapters value must be the same for all requests");
    }
    m_main_pipeline->set_adapters(sampling_params[0].adapters);
    m_draft_pipeline->set_adapters(sampling_params[0].adapters);

    const auto streamer_ptr = std::make_shared<ThreadedStreamerWrapper>(streamer, m_tokenizer);

    OPENVINO_ASSERT(!streamer_ptr->has_callback() || input_ids.size() == 1 && (sampling_params[0].is_greedy_decoding() || sampling_params[0].is_multinomial()),
        "Currently streaming is possible only with batch size=1 and only for greedy or multinomial decoding");

    std::vector<GenerationHandle> main_generations;
    for (size_t request_id = 0; request_id < input_ids.size(); ++request_id) {
        m_sd_metrics.set_generated_len(request_id, sampling_params[request_id].get_max_new_tokens(input_ids[request_id].get_size()));
        OPENVINO_ASSERT(1 == input_ids[request_id].get_shape().at(0), "Use multiple tensors to pass a batch.");
        main_generations.push_back(m_main_pipeline->add_request(request_id, input_ids[request_id], sampling_params[request_id]));

        auto draft_sampling_params = sampling_params[request_id];
        // set the parameters do not stop draft generation without stopping of the same request for main pipeline
        draft_sampling_params.ignore_eos = true;
        draft_sampling_params.stop_strings = {};
        std::lock_guard<std::mutex> lock(m_draft_generations_mutex);
        auto draft_generation = m_draft_generations.find(request_id);
        if (draft_generation != m_draft_generations.end()) {
            m_draft_generations.erase(draft_generation);
        }
        m_draft_generations.insert({request_id, m_draft_pipeline->add_request(request_id, input_ids[request_id], draft_sampling_params)});
    }
    auto all_requests = get_awaiting_requests();

    GenerationHandle& generation = main_generations.at(0);

    streamer_ptr->start();

    bool get_first_token = false;
    float first_token_time = 0;
    int first_tokens_num = 0;
    m_draft_pipeline->reset_infer_duration();
    m_main_pipeline->reset_infer_duration();
    while (has_non_finished_requests()) {
        try {
            ManualTimer step_timer("speculative_decoding: step()");
            step_timer.start();    
            step();
            step_timer.end();
            first_token_time += step_timer.get_duration();
            if (generation.get()->can_read()) {
                std::unordered_map<uint64_t, GenerationOutput> token = generation.get()->read();
                if (!get_first_token && !token.begin()->second.generated_ids.empty()) {
                    first_tokens_num = token.begin()->second.generated_ids.size();
                }    
            }
            if (!get_first_token && first_tokens_num > 0) {
                get_first_token = true;
                m_sd_metrics.first_token_duration = first_token_time;
                int number = 0;
                m_draft_pipeline->get_infer_duration(m_sd_metrics.draft_infer_for_first_token, number);
                m_main_pipeline->get_infer_duration(m_sd_metrics.main_infer_for_first_token, number);
            }
        } catch (...) {
            drop_requests(); // remove all requests from pipeline state in case of exception
            streamer_ptr->end();
            std::rethrow_exception(std::current_exception());
        }
        stream_tokens(streamer_ptr, generation);
    }

    // waiting for competion of streaming
    streamer_ptr->end();

    OPENVINO_ASSERT(is_requests_empty(), "Internal error: current request is supposed to be dropped within step() function as completed");

    std::vector<EncodedGenerationResult> results;
    results.reserve(all_requests.size());

    generate_timer.end();

    for (size_t request_id = 0; request_id < all_requests.size(); ++request_id) {
        const auto& request = all_requests[request_id];
        auto sampling_params = request->get_sampling_parameters();
        const auto& sequences = request->get_finished_sequences();
        size_t num_outputs = std::min(sampling_params.num_return_sequences, sequences.size());

        EncodedGenerationResult result;
        result.m_request_id = request_id;
        result.m_generation_ids.resize(num_outputs);
        result.m_scores.resize(num_outputs);
        result.m_status = request->get_generation_stream()->get_status();

        for (size_t i = 0; i < num_outputs; ++i) {
            const auto & sequence = sequences[i];
            const float score = sampling_params.is_beam_search() ? sequence->get_beam_search_score(sampling_params) : sequence->get_cumulative_log_prob();
            const auto & generated_ids = sequence->get_generated_ids();

            if (sampling_params.echo) {
                result.m_generation_ids[i] = request->get_prompt_ids();
            }
            std::copy(generated_ids.begin(), generated_ids.end(), std::back_inserter(result.m_generation_ids[i]));
            result.m_scores[i] = score;
        }

        result.m_status = main_generations[request_id]->get_status();

        // The same perf metrics for each sequence, only tokenization/detokenization will differ.
        m_perf_metrics.raw_metrics.generate_durations.clear();
        m_perf_metrics.raw_metrics.generate_durations.emplace_back(generate_timer.get_duration_microsec());
        m_perf_metrics.num_input_tokens = request->get_prompt_len();
        m_perf_metrics.evaluate_statistics(generate_timer.get_start_time());

        result.perf_metrics = m_perf_metrics;
        results.push_back(std::move(result));
    }

    OPENVINO_ASSERT(results.size() == input_ids.size());
    generate_timer.end();
    return results;
}

SpeculativeDecodingMetrics
ContinuousBatchingPipeline::SpeculativeDecodingImpl::get_speculative_decoding_metrics() {
    return m_sd_metrics;
};

void ContinuousBatchingPipeline::SpeculativeDecodingImpl::drop_requests() {
    m_draft_pipeline->finish_request();
    m_main_pipeline->finish_request();
}


bool ContinuousBatchingPipeline::SpeculativeDecodingImpl::is_requests_empty() {
    return m_main_pipeline->is_requests_empty() && m_draft_pipeline->is_requests_empty();
}

std::vector<SequenceGroup::Ptr> ContinuousBatchingPipeline::SpeculativeDecodingImpl::get_awaiting_requests() {
    auto main_awaiting_requests = m_main_pipeline->get_awaiting_requests();
    auto draft_awaiting_requests = m_draft_pipeline->get_awaiting_requests();
    OPENVINO_ASSERT(main_awaiting_requests.size() == draft_awaiting_requests.size());
    return main_awaiting_requests;
}
}
