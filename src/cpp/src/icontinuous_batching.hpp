// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/continuous_batching_pipeline.hpp"

#include "cache_manager.hpp"
#include "sampler.hpp"
#include "model_runner.hpp"
#include "scheduler.hpp"

namespace ov::genai {

/**
 * Base interface for all continuous batching based pipelines
 */
class ContinuousBatchingPipeline::IContinuousBatchingPipeline {
protected:
    Tokenizer m_tokenizer;

    // TODO (mzegla): GenerationConfig is request specific object
    // and pipeline only uses default rng_seed and some special tokens.
    ov::genai::GenerationConfig m_generation_config;

    PipelineMetrics m_pipeline_metrics;

    struct PerfTime {
        float m_paged_attention_time_ms = 0.0f;
        float m_matmul_time_ms = 0.0f;
        float m_infer_total_ms = 0.0f;

        ~PerfTime() {
            // std::cout << "Inference requests aggregated statistic: " << std::endl;
            // std::cout << "Paged attention % of inference execution: " << (m_paged_attention_time_ms / m_infer_total_ms) * 100 << std::endl;
            // std::cout << "MatMul % of inference execution: " << (m_matmul_time_ms / m_infer_total_ms) * 100 << std::endl;
            // std::cout << "Total inference execution secs: " << m_infer_total_ms / 1000. << std::endl;
            // std::cout << std::endl;
        }
    } m_perf;

    bool m_is_chat_conversation = false;
    ChatHistory m_history;

public:
    ov::genai::GenerationConfig get_config() const;
    PipelineMetrics get_metrics() const;
    ov::genai::Tokenizer get_tokenizer();

    /**
     * Adds requests to awaiting queue using encoded inputs
     */
    virtual GenerationHandle add_request(uint64_t request_id,
                                         const ov::Tensor& input_ids,
                                         ov::genai::GenerationConfig sampling_params) = 0;

    /**
     * Adds request to running queue based on string input
     * This step also performs tokenization's encode
     */
    virtual GenerationHandle add_request(uint64_t request_id,
                                         const std::string& prompt,
                                         ov::genai::GenerationConfig sampling_params) = 0;
    
    /**
     * Checks whether server (pipeline) has non-finished requests and step() should be called within a loop
     */
    virtual bool has_non_finished_requests() = 0;

    /**
     * Performs a single inference step of all running (and pulls awaiting) requests
     */
    virtual void step() = 0;

    /**
     * Performs monolitic generation based on encoded prompts
     */
    virtual std::vector<EncodedGenerationResult>
    generate(const std::vector<ov::Tensor>& input_ids,
             const std::vector<GenerationConfig>& sampling_params,
             const StreamerVariant& streamer) = 0;

    /**
     * Performs monolitic generation based on text prompts
     */
    std::vector<GenerationResult>
    generate(const std::vector<std::string>& prompts,
             std::vector<ov::genai::GenerationConfig> sampling_params,
             const StreamerVariant& streamer);

    /**
     * Starts chat with a given system prompt
     * 
     * In chat scenario prompts passed to `generate` method are accumulated inside the pipeline until `finish_chat` is called
     */
    void start_chat(const std::string& system_message);

    /**
     * Ends chat
     */
    void finish_chat();
};
}