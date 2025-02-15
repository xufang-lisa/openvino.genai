// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/image_generation/text2image_pipeline.hpp"
#include "openvino/genai/image_generation/image2image_pipeline.hpp"
#include "openvino/genai/image_generation/inpainting_pipeline.hpp"
#include <cxxopts.hpp>
#include <chrono>
#include "imwrite.hpp"
#include "load_image.hpp"
#include "progress_bar.hpp"

inline void print_one_generate(ov::genai::ImageGenerationPerfMetrics& metrics, std::string prefix, int idx) {
    std::string prefix_idx = "[" + prefix + "-" + std::to_string(idx) + "]";
    std::cout << "\n";
    std::cout << prefix_idx << " generate time: " << metrics.get_generate_duration()
              << " ms, total infer time:" << metrics.get_all_infer_duration()
              << " ms" << std::endl;
    std::cout << prefix_idx << " encoder infer time: " << metrics.get_encoder_infer_duration() << " ms"<< std::endl;
    float first_iter_time, other_iter_avg_time;
    float first_infer_time, other_infer_avg_time;
    metrics.get_iteration_duration(first_iter_time, other_iter_avg_time);
    if (!metrics.raw_metrics.transformer_inference_durations.empty()) {
        metrics.get_transformer_infer_duration(first_infer_time, other_infer_avg_time);
        std::cout << prefix_idx << " transformer iteration num:" << metrics.raw_metrics.iteration_durations.size()
                  << ", first iteration time:" << first_iter_time
                  << " ms, other iteration avg time:" << other_iter_avg_time << " ms" << std::endl;
        std::cout << prefix_idx
                  << " transformer inference num:" << metrics.raw_metrics.transformer_inference_durations.size()
                  << ", first inference time:" << first_infer_time
                  << " ms, other inference avg time:" << other_infer_avg_time << " ms" << std::endl;
    } else {
        metrics.get_unet_infer_duration(first_infer_time, other_infer_avg_time);
        std::cout << prefix_idx << " unet iteration num:" << metrics.raw_metrics.iteration_durations.size()
                  << ", first iteration time:" << first_iter_time
                  << " ms, other iteration avg time:" << other_iter_avg_time << " ms" << std::endl;
        std::cout << prefix_idx << " unet inference num:" << metrics.raw_metrics.unet_inference_durations.size()
                  << ", first inference time:" << first_infer_time
                  << " ms, other inference avg time:" << other_infer_avg_time << " ms" << std::endl;
    }
    std::cout << prefix_idx << " vae decoder infer time:" << metrics.vae_decoder_inference_duration << " ms"
              << std::endl;
}

inline float calculate_average(std::vector<float>& durations) {
    float duration_mean = std::accumulate(durations.begin(),
                                           durations.end(),
                                           0.0f,
                                           [](const float& acc, const float& duration) -> float {
                                               return acc + duration;
                                           });
    if (!durations.empty()) {
        duration_mean /= durations.size();
    }
    return duration_mean;
}

inline void print_statistic(std::vector<ov::genai::ImageGenerationPerfMetrics>& warmup_metrics, std::vector<ov::genai::ImageGenerationPerfMetrics>& iter_metrics) {
    std::vector<float> generate_durations;
    std::vector<float> total_inference_durations;
    std::vector<float> encoder_durations;
    std::vector<float> decoder_durations;
    float load_time = 0.0f;
    int warmup_num = warmup_metrics.size();
    int iter_num = iter_metrics.size();

    float generate_warmup = 0.0f;
    float inference_warmup = 0.0f;
    if (!warmup_metrics.empty()) {
        generate_warmup = warmup_metrics[0].get_generate_duration();
        inference_warmup = warmup_metrics[0].get_all_infer_duration();
    }

    for (auto& metrics : iter_metrics) {
        generate_durations.emplace_back(metrics.get_generate_duration());
        total_inference_durations.emplace_back(metrics.get_all_infer_duration());
        decoder_durations.emplace_back(metrics.get_decoder_infer_duration());
        encoder_durations.emplace_back(metrics.get_encoder_infer_duration());
        load_time = metrics.get_load_time();
    }

    float generate_mean = calculate_average(generate_durations);
    float inference_mean = calculate_average(total_inference_durations);
    float decoder_mean = calculate_average(decoder_durations);
    float encoder_mean = calculate_average(encoder_durations);

    std::cout << "\nTest finish, load time: " << load_time << " ms" << std::endl;
    std::cout << "Warmup number:" << warmup_num << ", first generate warmup time:" << generate_warmup
              << " ms, infer warmup time:" << inference_warmup << " ms" << std::endl;
    std::cout << "Generate iteration number:" << iter_num << ", for one iteration, generate avg time: " << generate_mean
              << " ms, infer avg time:" << inference_mean << " ms, total encoder infer avg time:" << encoder_mean
              << " ms, decoder infer avg time:" << decoder_mean << " ms" << std::endl;
}

void text2image(cxxopts::ParseResult& result) {
    std::string prompt = result["prompt"].as<std::string>();
    const std::string models_path = result["model"].as<std::string>();
    std::string device = result["device"].as<std::string>();
    size_t num_warmup = result["num_warmup"].as<size_t>();
    size_t num_iter = result["num_iter"].as<size_t>();
    const std::string output_dir = result["output_dir"].as<std::string>();

    ov::genai::Text2ImagePipeline pipe(models_path, device);
    ov::genai::ImageGenerationConfig config = pipe.get_generation_config();
    config.width = result["width"].as<size_t>();
    config.height = result["height"].as<size_t>();
    config.num_inference_steps = result["num_inference_steps"].as<size_t>();
    config.num_images_per_prompt = result["num_images_per_prompt"].as<size_t>();
    pipe.set_generation_config(config);

    std::cout << std::fixed << std::setprecision(2);
    std::vector<ov::genai::ImageGenerationPerfMetrics> warmup_metrics;
    for (size_t i = 0; i < num_warmup; i++) {
        pipe.generate(prompt);
        ov::genai::ImageGenerationPerfMetrics metrics = pipe.get_performance_metrics();
        warmup_metrics.emplace_back(metrics);
        print_one_generate(metrics, "warmup", i);
    }

    std::vector<ov::genai::ImageGenerationPerfMetrics> iter_metrics;
    for (size_t i = 0; i < num_iter; i++) {
        ov::Tensor image = pipe.generate(prompt);
        ov::genai::ImageGenerationPerfMetrics metrics = pipe.get_performance_metrics();
        iter_metrics.emplace_back(metrics);
        std::string image_name = output_dir + "/image_" + std::to_string(i) + ".bmp";
        imwrite(image_name, image, true);
        print_one_generate(metrics, "iter", i);
    }

    print_statistic(warmup_metrics, iter_metrics);
}

void image2image(cxxopts::ParseResult& result) {
    std::string prompt = result["prompt"].as<std::string>();
    const std::string models_path = result["model"].as<std::string>();
    std::string image_path = result["image"].as<std::string>();
    std::string device = result["device"].as<std::string>();
    size_t num_warmup = result["num_warmup"].as<size_t>();
    size_t num_iter = result["num_iter"].as<size_t>();
    const std::string output_dir = result["output_dir"].as<std::string>();
    float strength = result["strength"].as<float>();

    ov::Tensor image_input = utils::load_image(image_path);

    ov::genai::Image2ImagePipeline pipe(models_path, device);

    std::vector<ov::genai::ImageGenerationPerfMetrics> warmup_metrics;
    std::cout << std::fixed << std::setprecision(2);
    for (size_t i = 0; i < num_warmup; i++) {
        pipe.generate(prompt, image_input, ov::genai::strength(strength), ov::genai::callback(progress_bar));
        ov::genai::ImageGenerationPerfMetrics metrics = pipe.get_performance_metrics();
        warmup_metrics.emplace_back(metrics);
        print_one_generate(metrics, "warmup", i);
    }

    std::vector<ov::genai::ImageGenerationPerfMetrics> iter_metrics;
    for (size_t i = 0; i < num_iter; i++) {
        ov::Tensor image = pipe.generate(prompt, image_input, ov::genai::strength(strength), ov::genai::callback(progress_bar));
        ov::genai::ImageGenerationPerfMetrics metrics = pipe.get_performance_metrics();
        iter_metrics.emplace_back(metrics);
        std::string image_name = output_dir + "/image_" + std::to_string(i) + ".bmp";
        imwrite(image_name, image, true);
        print_one_generate(metrics, "iter", i);
    }

    print_statistic(warmup_metrics, iter_metrics);
}

void inpainting(cxxopts::ParseResult& result) {
    std::string prompt = result["prompt"].as<std::string>();
    const std::string models_path = result["model"].as<std::string>();
    std::string image_path = result["image"].as<std::string>();
    std::string mask_image_path = result["mask_image"].as<std::string>();
    std::string device = result["device"].as<std::string>();
    size_t num_warmup = result["num_warmup"].as<size_t>();
    size_t num_iter = result["num_iter"].as<size_t>();
    const std::string output_dir = result["output_dir"].as<std::string>();

    ov::Tensor image_input = utils::load_image(image_path);
    ov::Tensor mask_image = utils::load_image(mask_image_path);

    ov::genai::InpaintingPipeline pipe(models_path, device);

    std::cout << std::fixed << std::setprecision(2);
    std::vector<ov::genai::ImageGenerationPerfMetrics> warmup_metrics;
    for (size_t i = 0; i < num_warmup; i++) {
        pipe.generate(prompt, image_input, mask_image, ov::genai::callback(progress_bar));
        ov::genai::ImageGenerationPerfMetrics metrics = pipe.get_performance_metrics();
        warmup_metrics.emplace_back(metrics);
        print_one_generate(metrics, "warmup", i);
    }

    std::vector<ov::genai::ImageGenerationPerfMetrics> iter_metrics;
    for (size_t i = 0; i < num_iter; i++) {
        ov::Tensor image = pipe.generate(prompt, image_input, mask_image, ov::genai::callback(progress_bar));
        ov::genai::ImageGenerationPerfMetrics metrics = pipe.get_performance_metrics();
        iter_metrics.emplace_back(metrics);
        std::string image_name = output_dir + "/image_" + std::to_string(i) + ".bmp";
        imwrite(image_name, image, true);
        print_one_generate(metrics, "iter", i);
    }

    print_statistic(warmup_metrics, iter_metrics);
}

int main(int argc, char* argv[]) try {
    cxxopts::Options options("benchmark_image_generation", "Help command");

    options.add_options()
    //common parameters
    ("t,pipeline_type", "pipeline type: text2image/image2image/inpainting", cxxopts::value<std::string>()->default_value("text2image"))
    ("m,model", "Path to model and tokenizers base directory", cxxopts::value<std::string>())
    ("p,prompt", "Prompt", cxxopts::value<std::string>()->default_value("The Sky is blue because"))
    ("nw,num_warmup", "Number of warmup iterations", cxxopts::value<size_t>()->default_value(std::to_string(1)))
    ("n,num_iter", "Number of iterations", cxxopts::value<size_t>()->default_value(std::to_string(3)))
    ("d,device", "device", cxxopts::value<std::string>()->default_value("CPU"))
    ("o,output_dir", "Path to save output image", cxxopts::value<std::string>()->default_value("."))
    ("is,num_inference_steps", "The number of inference steps used to denoise initial noised latent to final image", cxxopts::value<size_t>()->default_value(std::to_string(20)))
    ("ni,num_images_per_prompt", "The number of images to generate per generate() call", cxxopts::value<size_t>()->default_value(std::to_string(1)))
    ("i,image", "Image path", cxxopts::value<std::string>())
    //special parameters of text2image pipeline
    ("wh,width", "The width of the resulting image", cxxopts::value<size_t>()->default_value(std::to_string(512)))
    ("ht,height", "The height of the resulting image", cxxopts::value<size_t>()->default_value(std::to_string(512)))
    //special parameters of image2image pipeline
    ("s,strength", "Indicates extent to transform the reference `image`. Must be between 0 and 1", cxxopts::value<float>()->default_value(std::to_string(0.8)))
    //special parameters of inpainting pipeline
    ("mi,mask_image", "Mask image path", cxxopts::value<std::string>())
    ("h,help", "Print usage");

    cxxopts::ParseResult result;
    try {
        result = options.parse(argc, argv);
    } catch (const cxxopts::exceptions::exception& e) {
        std::cout << e.what() << "\n\n";
        std::cout << options.help() << std::endl;
        return EXIT_FAILURE;
    }

    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        return EXIT_SUCCESS;
    }

    std::string pipeline_type = result["pipeline_type"].as<std::string>();
    if (pipeline_type == "text2image") {
        text2image(result);
    } else if (pipeline_type == "image2image") {
        image2image(result);
    } else if (pipeline_type == "inpainting") {
        inpainting(result);
    } else {
        std::cout << "not support pipeline type: " << pipeline_type << std::endl;
    }

    return 0;
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
