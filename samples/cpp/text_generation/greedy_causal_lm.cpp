// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/llm_pipeline.hpp"
#include "read_prompt_from_file.h"

int main(int argc, char* argv[]) try {
    if (3 > argc)
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <MODEL_DIR> \"<PROMPT>\"");

    std::string models_path = argv[1];
    std::string prompt = argv[2];
    std::string device = "GPU";  // GPU can be used as well
    ov::genai::LLMPipeline pipe(models_path, device);
    ov::genai::GenerationConfig config;
    if (std::filesystem::is_regular_file(prompt)) {
        std::string prompt_file = prompt;
        prompt = utils::read_prompt(prompt_file);
    }
    config.max_new_tokens = 129;
    auto start_time = std::chrono::high_resolution_clock::now();
    auto streamer = [](std::string subword) {
        std::cout << subword << std::flush;
        return ov::genai::StreamingStatus::RUNNING;
    };
    auto result = pipe.generate(prompt, config, streamer);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "\nGeneration completed in " << duration.count() << " ms" << std::endl;
    ov::genai::PerfMetrics metrics = result.perf_metrics;
    auto generation_duration = metrics.get_generate_duration().mean;
    auto inference_duration = metrics.get_inference_duration().mean;
    auto sample_duration = metrics.get_sample_duration().mean;
    auto scheduler_duration = metrics.get_schedule_duration().mean;
    std::cout << "Generate time: " << generation_duration << " ms" << std::endl;
    std::cout << "Inference+sample+scheduler time: " << inference_duration << " ms("
            << inference_duration * 100.0 / generation_duration << "%) + " << sample_duration << " ms("
            << sample_duration * 100.0 / generation_duration << "%) + " << scheduler_duration
            << " ms(" << scheduler_duration * 100.0 / generation_duration << "%)" << std::endl;
    std::cout << "Output token size:" << metrics.get_num_generated_tokens() << std::endl;
    std::cout << "TTFT: " << metrics.get_ttft().mean  << " ± " << metrics.get_ttft().std << " ms" << std::endl;
    std::cout << "TPOT: " << metrics.get_tpot().mean  << " ± " << metrics.get_tpot().std << " ms/token " << std::endl;
    std::cout << "Total iteration number: " << metrics.raw_metrics.m_token_infer_durations.size() << std::endl;
    std::cout << "Input token size: " << metrics.get_num_input_tokens() << std::endl;
    // std::cout << result << std::endl;
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
