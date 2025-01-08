// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <ctime>
#include <cstdlib>
#include <filesystem>

#include "openvino/genai/image_generation/inpainting_pipeline.hpp"
#include "openvino/genai/image_generation/image2image_pipeline.hpp"

#include "image_generation/stable_diffusion_pipeline.hpp"
#include "image_generation/stable_diffusion_xl_pipeline.hpp"

#include "utils.hpp"

namespace ov {
namespace genai {

InpaintingPipeline::InpaintingPipeline(const std::filesystem::path& root_dir) {
    const std::string class_name = get_class_name(root_dir);

    if (class_name == "StableDiffusionPipeline" || 
        class_name == "LatentConsistencyModelPipeline" ||
        class_name == "StableDiffusionInpaintPipeline") {
        m_impl = std::make_shared<StableDiffusionPipeline>(PipelineType::INPAINTING, root_dir);
    } else if (class_name == "StableDiffusionXLPipeline" || class_name == "StableDiffusionXLInpaintPipeline") {
        m_impl = std::make_shared<StableDiffusionXLPipeline>(PipelineType::INPAINTING, root_dir);
    } else {
        OPENVINO_THROW("Unsupported text to image generation pipeline '", class_name, "'");
    }
}

InpaintingPipeline::InpaintingPipeline(const std::filesystem::path& root_dir, const std::string& device, const ov::AnyMap& properties) {
    const std::string class_name = get_class_name(root_dir);

    if (class_name == "StableDiffusionPipeline" ||
        class_name == "LatentConsistencyModelPipeline" ||
        class_name == "StableDiffusionInpaintPipeline") {
        m_impl = std::make_shared<StableDiffusionPipeline>(PipelineType::INPAINTING, root_dir, device, properties);
    } else if (class_name == "StableDiffusionXLPipeline" || class_name == "StableDiffusionXLInpaintPipeline") {
        m_impl = std::make_shared<StableDiffusionXLPipeline>(PipelineType::INPAINTING, root_dir, device, properties);
    } else {
        OPENVINO_THROW("Unsupported text to image generation pipeline '", class_name, "'");
    }
}

InpaintingPipeline::InpaintingPipeline(const Image2ImagePipeline& pipe) {
    if (auto stable_diffusion_xl = std::dynamic_pointer_cast<StableDiffusionXLPipeline>(pipe.m_impl); stable_diffusion_xl != nullptr) {
        m_impl = std::make_shared<StableDiffusionXLPipeline>(PipelineType::INPAINTING, *stable_diffusion_xl);
    } else if (auto stable_diffusion = std::dynamic_pointer_cast<StableDiffusionPipeline>(pipe.m_impl); stable_diffusion != nullptr) {
        m_impl = std::make_shared<StableDiffusionPipeline>(PipelineType::INPAINTING, *stable_diffusion);
    } else {
        OPENVINO_ASSERT("Cannot convert specified Image2ImagePipeline to InpaintingPipeline");
    }
}

InpaintingPipeline::InpaintingPipeline(const std::shared_ptr<DiffusionPipeline>& impl)
    : m_impl(impl) {
    assert(m_impl != nullptr);
}

InpaintingPipeline InpaintingPipeline::stable_diffusion(
    const std::shared_ptr<Scheduler>& scheduler,
    const CLIPTextModel& clip_text_model,
    const UNet2DConditionModel& unet,
    const AutoencoderKL& vae) {
    auto impl = std::make_shared<StableDiffusionPipeline>(PipelineType::INPAINTING, clip_text_model, unet, vae);

    assert(scheduler != nullptr);
    impl->set_scheduler(scheduler);

    return InpaintingPipeline(impl);
}

InpaintingPipeline InpaintingPipeline::latent_consistency_model(
    const std::shared_ptr<Scheduler>& scheduler,
    const CLIPTextModel& clip_text_model,
    const UNet2DConditionModel& unet,
    const AutoencoderKL& vae) {
    auto impl = std::make_shared<StableDiffusionPipeline>(PipelineType::INPAINTING, clip_text_model, unet, vae);

    assert(scheduler != nullptr);
    impl->set_scheduler(scheduler);

    return InpaintingPipeline(impl);
}

InpaintingPipeline InpaintingPipeline::stable_diffusion_xl(
    const std::shared_ptr<Scheduler>& scheduler,
    const CLIPTextModel& clip_text_model,
    const CLIPTextModelWithProjection& clip_text_model_with_projection,
    const UNet2DConditionModel& unet,
    const AutoencoderKL& vae) {
    auto impl = std::make_shared<StableDiffusionXLPipeline>(PipelineType::INPAINTING, clip_text_model, clip_text_model_with_projection, unet, vae);

    assert(scheduler != nullptr);
    impl->set_scheduler(scheduler);

    return InpaintingPipeline(impl);
}

ImageGenerationConfig InpaintingPipeline::get_generation_config() const {
    return m_impl->get_generation_config();
}

void InpaintingPipeline::set_generation_config(const ImageGenerationConfig& generation_config) {
    m_impl->set_generation_config(generation_config);
}

void InpaintingPipeline::set_scheduler(std::shared_ptr<Scheduler> scheduler) {
    m_impl->set_scheduler(scheduler);
}

void InpaintingPipeline::reshape(const int num_images_per_prompt, const int height, const int width, const float guidance_scale) {
    m_impl->reshape(num_images_per_prompt, height, width, guidance_scale);
}

void InpaintingPipeline::compile(const std::string& device, const ov::AnyMap& properties) {
    m_impl->compile(device, properties);
}

ov::Tensor InpaintingPipeline::generate(const std::string& positive_prompt, ov::Tensor initial_image, ov::Tensor mask, const ov::AnyMap& properties) {
    OPENVINO_ASSERT(initial_image, "Initial image cannot be empty when passed to InpaintingPipeline::generate");
    OPENVINO_ASSERT(mask, "Mask image cannot be empty when passed to InpaintingPipeline::generate");
    return m_impl->generate(positive_prompt, initial_image, mask, properties);
}

ov::Tensor InpaintingPipeline::decode(const ov::Tensor latent) {
    float infer_duration;
    return m_impl->decode(latent, infer_duration);
}

ImageGenerationPerfMetrics InpaintingPipeline::get_perfomance_metrics() {
    return m_impl->get_perfomance_metrics();
}

}  // namespace genai
}  // namespace ov
