// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief This is a header file for OpenVINO GenAI C API, which is a C wrapper for  ov::genai::LLMPipeline class.
 *
 * @file llm_pipeline_c.h
 */

#pragma once
#include "generation_config.h"
#include "perf_metrics.h"

/**
 * @struct ov_genai_decoded_results
 * @brief type define ov_genai_decoded_results from ov_genai_decoded_results_opaque
 */
typedef struct ov_genai_decoded_results_opaque ov_genai_decoded_results;

/**
 * @brief Create DecodedResults
 * @param results A pointer to the newly created ov_genai_decoded_results.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_decoded_results_create(ov_genai_decoded_results** results);

/**
 * @brief Release the memory allocated by ov_genai_decoded_results.
 * @param model A pointer to the ov_genai_decoded_results to free memory.
 */
OPENVINO_GENAI_C_EXPORTS void ov_genai_decoded_results_free(ov_genai_decoded_results* results);

/**
 * @brief Get performance metrics from ov_genai_decoded_results.
 * @param results A pointer to the ov_genai_decoded_results instance.
 * @param metrics A pointer to the newly created ov_genai_perf_metrics.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_decoded_results_get_perf_metrics(const ov_genai_decoded_results* results,
                                                                               ov_genai_perf_metrics** metrics);

/**
 * @brief Release the memory allocated by ov_genai_perf_metrics.
 * @param model A pointer to the ov_genai_perf_metrics to free memory.
 */
OPENVINO_GENAI_C_EXPORTS void ov_genai_decoded_results_perf_metrics_free(ov_genai_perf_metrics* metrics);

/**
 * @brief Get string result from ov_genai_decoded_results.
 * @param results A pointer to the ov_genai_decoded_results instance.
 * @param output A pointer to the pre-allocated output string buffer. It can be set to NULL, in which case the
 * *output_size will provide the needed buffer size. The user should then allocate the required buffer size and call
 * this function again to obtain the entire output.
 * @param output_size A Pointer to the size of the output string from the results, including the null terminator. If
 * output is not NULL, *output_size should be greater than or equal to the result string size; otherwise, the function
 * will return OUT_OF_BOUNDS(-6).
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_decoded_results_get_string(const ov_genai_decoded_results* results,
                                                                         char* output,
                                                                         size_t* output_size);

/**
 * @struct ov_genai_llm_pipeline
 * @brief type define ov_genai_llm_pipeline from ov_genai_llm_pipeline_opaque
 * @return ov_status_e A status code, return OK(0) if successful.
 */
typedef struct ov_genai_llm_pipeline_opaque ov_genai_llm_pipeline;

/**
 * @brief Construct ov_genai_llm_pipeline.
 * @param models_path Path to the directory containing the model files.
 * @param device Name of a device to load a model to.
 * @param ov_genai_llm_pipeline A pointer to the newly created ov_genai_llm_pipeline.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_llm_pipeline_create(const char* models_path,
                                                                  const char* device,
                                                                  ov_genai_llm_pipeline** pipe);

// TODO: Add 'const ov::AnyMap& properties' as an input argument when creating ov_genai_llm_pipeline.

/**
 * @brief Release the memory allocated by ov_genai_llm_pipeline.
 * @param model A pointer to the ov_genai_llm_pipeline to free memory.
 */
OPENVINO_GENAI_C_EXPORTS void ov_genai_llm_pipeline_free(ov_genai_llm_pipeline* pipe);

typedef enum {
    OV_GENAI_STREAMMING_STATUS_RUNNING = 0,  // Continue to run inference
    OV_GENAI_STREAMMING_STATUS_STOP =
        1,  // Stop generation, keep history as is, KV cache includes last request and generated tokens
    OV_GENAI_STREAMMING_STATUS_CANCEL = 2  // Stop generate, drop last prompt and all generated tokens from history, KV
                                           // cache includes history but last step
} ov_genai_streamming_status_e;

/**
 * @brief Structure for streamer callback functions with arguments.
 *
 * The callback function takes two parameters:
 * - `const char* str`: A constant string extracted from the decoded result for processing
 * - `void* args`: A pointer to additional arguments, allowing flexible data passing.
 */
typedef struct {
    ov_genai_streamming_status_e(
        OPENVINO_C_API_CALLBACK* callback_func)(const char* str, void* args);  //!< Pointer to the callback function
    void* args;  //!< Pointer to the arguments passed to the callback function
} streamer_callback;

/**
 * @brief Generate results by ov_genai_llm_pipeline
 * @param pipe A pointer to the ov_genai_llm_pipeline instance.
 * @param inputs A pointer to the input string.
 * @param config A pointer to the ov_genai_generation_config, the pointer can be NULL.
 * @param streamer A pointer to the stream callback. Set to NULL if no callback is needed. Either this or results must
 * be non-NULL.
 * @param results A pointer to the ov_genai_decoded_results, which retrieves the results of the generation. Either this
 * or streamer must be non-NULL.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_llm_pipeline_generate(ov_genai_llm_pipeline* pipe,
                                                                    const char* inputs,
                                                                    const ov_genai_generation_config* config,
                                                                    const streamer_callback* streamer,
                                                                    ov_genai_decoded_results** results);
/**
 * @brief Start chat with keeping history in kv cache.
 * @param pipe A pointer to the ov_genai_llm_pipeline instance.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_llm_pipeline_start_chat(ov_genai_llm_pipeline* pipe);

/**
 * @brief Finish chat and clear kv cache.
 * @param pipe A pointer to the ov_genai_llm_pipeline instance.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_llm_pipeline_finish_chat(ov_genai_llm_pipeline* pipe);

/**
 * @brief Get the GenerationConfig from ov_genai_llm_pipeline.
 * @param pipe A pointer to the ov_genai_llm_pipeline instance.
 * @param ov_genai_generation_config A pointer to the newly created ov_genai_generation_config.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_llm_pipeline_get_generation_config(const ov_genai_llm_pipeline* pipe,
                                                                                 ov_genai_generation_config** config);

/**
 * @brief Set the GenerationConfig to ov_genai_llm_pipeline.
 * @param pipe A pointer to the ov_genai_llm_pipeline instance.
 * @param config A pointer to the ov_genai_generation_config instance.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_llm_pipeline_set_generation_config(ov_genai_llm_pipeline* pipe,
                                                                                 ov_genai_generation_config* config);
