// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/image_generation/text2image_pipeline.hpp"

#include "imwrite.hpp"

int32_t main(int32_t argc, char* argv[]) try {
    OPENVINO_ASSERT(argc == 3, "Usage: ", argv[0], " <MODEL_DIR> '<PROMPT>'");

    const std::string models_path = argv[1], prompt = argv[2];
    const std::string device = "CPU";  // GPU can be used as well

    ov::genai::Text2ImagePipeline pipe(models_path, device);
    auto image_results = pipe.generate(prompt,
        ov::genai::width(512),
        ov::genai::height(512),
        ov::genai::num_inference_steps(20),
        ov::genai::num_images_per_prompt(1));

    // writes `num_images_per_prompt` images by pattern name
    imwrite("image_%d.bmp", image_results.image, true);

    std::cout << "pipeline generate duration ms:" << image_results.perf_metrics.get_generate_duration().mean << std::endl;
    std::cout << "pipeline inference duration ms:" << image_results.perf_metrics.get_inference_duration().mean << std::endl;

    return EXIT_SUCCESS;
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
