// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <filesystem>
#include <openvino/openvino.hpp>

#include "openvino/genai/llm_pipeline.hpp"
#include "read_prompt_from_file.h"
#include "openvino/genai/speculative_decoding/perf_metrics.hpp"


void print_perf_metrics(ov::genai::SDPerfMetrics& perf_metrics, std::string model_name) {
    std::cout << "\n" << model_name << std::endl;
    auto generation_duration = perf_metrics.get_generate_duration().mean;
    auto inference_duration = perf_metrics.get_inference_duration().mean;
    auto sample_duration = perf_metrics.get_sample_duration().mean;
    auto scheduler_duration = perf_metrics.get_schedule_duration().mean;
    std::cout << "  Generate time: " << generation_duration << " ms" << std::endl;
    std::cout << "  Inference+sample+scheduler time: " << inference_duration << " ms("
              << inference_duration * 100.0 / generation_duration << "%) + " << sample_duration << " ms("
              << sample_duration * 100.0 / generation_duration << "%) + " << scheduler_duration
              << " ms(" << scheduler_duration * 100.0 / generation_duration << "%)" << std::endl;
    std::cout << "  TTFT: " << perf_metrics.get_ttft().mean << " ± " << perf_metrics.get_ttft().std
                << " ms" << std::endl;
    std::cout << "  TTST: " << perf_metrics.get_ttst().mean  << " ± " << perf_metrics.get_ttst().std << " ms/token " << std::endl;
    std::cout << "  TPOT: " << perf_metrics.get_tpot().mean  << " ± " << perf_metrics.get_tpot().std << " ms/iteration " << std::endl;
    std::cout << "  AVG Latency: " << perf_metrics.get_latency().mean  << " ± " << perf_metrics.get_latency().std << " ms/token " << std::endl;
    std::cout << "  Num generated token: " << perf_metrics.get_num_generated_tokens() << " tokens" << std::endl;
    std::cout << "  Total iteration number: " << perf_metrics.raw_metrics.m_durations.size() << std::endl;
}

int main(int argc, char* argv[]) try {
    if (4 != argc) {
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <MODEL_DIR> <EAGLE_MODEL_DIR> '<PROMPT>'");
    }

    std::string main_model_path = argv[1];
    std::string eagle_model_path = argv[2];
    std::string prompt = argv[3];
    if (std::filesystem::is_regular_file(prompt)) {
        std::string prompt_file = prompt;
        prompt = utils::read_prompt(prompt_file);
    }

    // Configure devices - can run main and eagle models on different devices
    std::string main_device = "GPU", eagle_device = "GPU"; // currently only GPU is used during developing

    // Eagle Speculative settings
    ov::genai::GenerationConfig config = ov::genai::greedy();
    config.max_new_tokens = 129;
    // Eagle specific parameters
    config.eagle_tree_params.branching_factor = 8; // Number of candidate tokens to consider at each level
    config.eagle_tree_params.tree_depth = 3; // How deep to explore the token tree
    config.eagle_tree_params.total_tokens = 16; // Total number of tokens to generate in eagle tree
    config.num_return_sequences = 1; // only support 1

    //config.eagle_tree_width = 3;    // Number of candidate tokens to consider at each level
    //config.eagle_tree_depth = 4;    // How deep to explore the token tree

    // Create pipeline with eagle speculative enabled
    ov::genai::LLMPipeline pipe(
        main_model_path,
        main_device,
        ov::genai::draft_model(eagle_model_path, eagle_device),
        std::pair<std::string, ov::Any>("eagle_mode", ov::Any("EAGLE3"))  // Specify eagle3 mode for draft model
    );
    // Setup performance measurement
    auto start_time = std::chrono::high_resolution_clock::now();

    // Optional: Create a streaming callback for real-time token display
    auto streamer = [](std::string subword) {
        std::cout << subword << std::flush;
        return ov::genai::StreamingStatus::RUNNING;
    };

    // Run generation with eagle speculative decoding
    std::cout << "Generating with Eagle Speculative decoding:" << std::endl;
    auto result = pipe.generate(prompt, config, streamer);
    std::cout << std::endl;

    // Calculate and display performance metrics
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "\nGeneration completed in " << duration.count() << " ms" << std::endl;

    auto sd_perf_metrics = std::dynamic_pointer_cast<ov::genai::SDPerModelsPerfMetrics>(result.extended_perf_metrics);
    if (sd_perf_metrics) {
        print_perf_metrics(sd_perf_metrics->main_model_metrics, "MAIN MODEL");
        std::cout << "  accepted token: " << sd_perf_metrics->get_num_accepted_tokens() << " tokens" << std::endl;
        std::cout << "  compress rate: "
                  << sd_perf_metrics->main_model_metrics.get_num_generated_tokens() * 1.0f /
                         sd_perf_metrics->main_model_metrics.raw_metrics.m_durations.size()
                  << std::endl;
        print_perf_metrics(sd_perf_metrics->draft_model_metrics, "DRAFT MODEL");
    }
    std::cout << std::endl;

    // Run without Eagle for comparison
    // std::cout << "\n-----------------------------" << std::endl;
    // std::cout << "Generating without Eagle Speculative decoding:" << std::endl;

    // Disable Eagle mode
    /*config.eagle_model = false;
    
    start_time = std::chrono::high_resolution_clock::now();
    pipe.generate(prompt, config, streamer);
    std::cout << std::endl;
    */
    // end_time = std::chrono::high_resolution_clock::now();
    // duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    // std::cout << "\nStandard generation completed in " << duration.count() << " ms" << std::endl;

} catch (const std::exception& error) {
    try {
        std::cerr << error.what() << '\n';
    } catch (const std::ios_base::failure&) {}
    return EXIT_FAILURE;
} catch (...) {
    try {
        std::cerr << "Non-exception object thrown\n";
    } catch (const std::ios_base::failure&) {}
    return EXIT_FAILURE;
}