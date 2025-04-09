// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "make_tokenizer_stateful.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/op/assign.hpp"


using namespace ov;
using namespace ov::op;

bool ov::genai::MakeAddSpecialTokensSatateful::run_on_model(const std::shared_ptr<ov::Model>& model) {
    std::shared_ptr<ov::Node> combine_seg_node;
    for (auto node: model->get_ordered_ops()) {
        if (strcmp(node->get_type_info().name, "CombineSegments") == 0) {
            combine_seg_node = node;
        }
    }
    if (!combine_seg_node) { 
        return false; 
    }
    
    size_t num_segments = (combine_seg_node->get_input_size() - 1) / 3;
    std::vector<Input<Node>> const_inputs;
    const_inputs.reserve(num_segments);

    for (size_t i = 0; i < num_segments; i++) {
        // If input is constant then it's special tokens, otherwise it's tokens from input text.
        auto const_input = std::dynamic_pointer_cast<v0::Constant>(combine_seg_node->get_input_node_shared_ptr(3*i + 1));
        if (const_input) { 
            const_inputs.emplace_back(combine_seg_node->input(3*i + 1));
        }
    }
    if (const_inputs.empty()) { 
        return false; 
    }

    // Default mode is add_special_tokens.
    auto default_mode_const = std::make_shared<v0::Constant>(ov::element::boolean, ov::Shape{}, std::vector{true});
    auto variable = std::make_shared<op::util::Variable>(op::util::VariableInfo{Shape{}, element::boolean, ADD_SPECIAL_TOKENS_VAR_ID});
    auto read_value = std::make_shared<v6::ReadValue>(default_mode_const, variable);
    auto zero_constant = std::make_shared<v0::Constant>(ov::element::i32, ov::Shape{}, std::vector{0});

    for (size_t i = 0; i < const_inputs.size(); i++) {
        auto select_node = std::make_shared<v1::Select>(read_value, const_inputs[i].get_source_output(), zero_constant);
        const_inputs[i].replace_source_output(select_node);
    }

    auto assign = std::make_shared<v6::Assign>(read_value, variable);
    model->add_sinks({assign});
    model->add_variables({variable});
    return true;
}


bool ov::genai::MakeVocabDecoderSatateful::run_on_model(const std::shared_ptr<ov::Model>& model) {
    std::shared_ptr<ov::Node> vocab_decoder_node;
    for (auto node: model->get_ordered_ops()) {
        if (strcmp(node->get_type_info().name, "VocabDecoder") == 0)
            vocab_decoder_node = node;
    }

    if (!vocab_decoder_node || vocab_decoder_node->get_input_size() < 5)
        return false;
    if (!vocab_decoder_node->input_value(4).get_element_type().is_integral_number())
        return false;
    
    std::shared_ptr<v0::Constant> skip_tokens_const = std::dynamic_pointer_cast<v0::Constant>(vocab_decoder_node->get_input_node_shared_ptr(4));
    std::shared_ptr<v8::Slice> skip_tokens_slice = std::dynamic_pointer_cast<v8::Slice>(vocab_decoder_node->get_input_node_shared_ptr(4));
    if (!skip_tokens_const && !skip_tokens_slice)
        return false;

    auto start_const = std::make_shared<v0::Constant>(ov::element::i32, ov::Shape{1}, std::vector{0});
    auto int_max_const = std::make_shared<v0::Constant>(ov::element::i32, ov::Shape{1}, std::vector{std::numeric_limits<int>::max()});
    auto one_const = std::make_shared<v0::Constant>(ov::element::i32, ov::Shape{1}, std::vector{1});
    
    // By default, INT_MAX will multiply with 1 and all skip_tokens will be selected.
    op::util::VariableInfo var_info{ov::Shape{1}, ov::element::i32, SKIP_SPECIAL_TOKENS_VAR_ID};
    auto variable = std::make_shared<op::util::Variable>(var_info);
    auto read_value = std::make_shared<v6::ReadValue>(one_const, variable);
    // if flag is set, then slice up to the int_max which means skip all tokens.
    auto stop = std::make_shared<v1::Multiply>(int_max_const, read_value);

    // If already has slice just replace the stop input.
    if (skip_tokens_slice) {
        skip_tokens_slice->input(2).replace_source_output(stop);
    } else {
        std::shared_ptr<v8::Slice> slice_node = std::make_shared<v8::Slice>(skip_tokens_const, start_const, stop, one_const);
        vocab_decoder_node->input(4).replace_source_output(slice_node->output(0));
    }
    
    auto assign = std::make_shared<v6::Assign>(read_value, variable);
    model->add_sinks({assign});
    model->add_variables({variable});
    return true;
}


bool ov::genai::MakePaddingSatateful::run_on_model(const std::shared_ptr<ov::Model>& model) {
    std::shared_ptr<ov::Node> combine_seg_node;
    for (auto node: model->get_ordered_ops()) {
        if (strcmp(node->get_type_info().name, "CombineSegments") == 0) {
            combine_seg_node = node;
        }
    }
    if (!combine_seg_node) { return false; }
    auto num_comb = combine_seg_node->get_input_size();
    
    size_t num_segments = (combine_seg_node->get_input_size() - 1) / 3;
    size_t number_of_main_tokens_inputs = 0;
    std::shared_ptr<Node> add_or_sub_node;
    for (size_t i = 0; i < num_segments; i++) {
        // Check all ends inputs of CombineSegments node.
        // For special tokens they are Constant/Select, 
        // for the ends input with main tokens sequence it's Add/Subtract.
        // If  Add then it's a right truncation, if Subtract then it's a left truncation.
        // For left truncation subtract is inserted on 0th input.
        auto tmp_sub_node = combine_seg_node->input_value(3*i + 0).get_node_shared_ptr();
        auto tmp_add_node = combine_seg_node->input_value(3*i + 1).get_node_shared_ptr();
        if (ov::as_type_ptr<v1::Add>(tmp_add_node)) {
            number_of_main_tokens_inputs += 1;
            add_or_sub_node = tmp_add_node;
        } else if (ov::as_type_ptr<v1::Subtract>(tmp_sub_node)) {
            number_of_main_tokens_inputs += 1;
            add_or_sub_node = tmp_sub_node;
        }
    }
    
    // Exit if couldn't find main input or there are several.
    if (number_of_main_tokens_inputs != 1) { return false; }

    // Minimum between max_length and length of token sequence.
    auto min_node = ov::as_type_ptr<v1::Minimum>(add_or_sub_node->get_input_node_shared_ptr(1));
    if (!min_node) { return false; }
    
    // constant containing final max_length - num_added tokens at the end of pipeline.
    auto const_node = ov::as_type_ptr<v0::Constant>(min_node->get_input_node_shared_ptr(1));
    if (!const_node) { return false; }

    op::util::VariableInfo var_info{const_node->get_output_shape(0), const_node->get_output_element_type(0), MAX_LENGTH_VAR_ID};
    auto max_length_var = std::make_shared<op::util::Variable>(var_info);

    size_t num_added_tokens = num_segments - number_of_main_tokens_inputs;
    // Constant which stores number of added_tokens.
    auto num_added_tokens_const = std::make_shared<v0::Constant>(
        const_node->get_output_element_type(0), const_node->get_output_shape(0), std::vector{num_added_tokens});
    
    OPENVINO_ASSERT(const_node->get_element_type() == element::i32);
    auto values = const_node->get_vector<int32_t>();
    OPENVINO_ASSERT(values.size() == 1);
    // Since const_node contain value = max_length - num_added tokens, 
    size_t default_max_length = values[0] + num_added_tokens;

    auto default_max_length_const = std::make_shared<v0::Constant>(
        const_node->get_output_element_type(0), const_node->get_output_shape(0), std::vector{default_max_length});

    // Save targets before adding new target with ReadValue to avoid recursion.
    auto target_inputs = const_node->output(0).get_target_inputs();
    auto max_length_rv = std::make_shared<v6::ReadValue>(default_max_length_const, max_length_var);
    auto subtract_node = std::make_shared<v1::Subtract>(max_length_rv, num_added_tokens_const);
    
    for (auto target_input : target_inputs) {
        target_input.replace_source_output(subtract_node->output(0));
    }

    // We need to check if user requested to not add special tokens.
    std::shared_ptr<v6::ReadValue> read_value_spec_tokens;
    for (const auto& sink : model->get_sinks()) {
        // Check if sink accepts input from Assign, and if that't the case get the ReadValus node input.
        if (auto read_value = ov::as_type_ptr<v6::ReadValue>(sink->get_input_node_shared_ptr(0))) {
            if (read_value->get_variable()->get_info().variable_id == ADD_SPECIAL_TOKENS_VAR_ID) {
                read_value_spec_tokens = read_value;
                break;
            }
        }
    }

    // If user requested to not add special tokens in order to correctly calculate 
    // truncation we need to enforce num_added_tokens to 0 regardless the hardcoded value of Constant.
    if (read_value_spec_tokens && num_added_tokens_const) {
        auto zero_constant = std::make_shared<v0::Constant>(ov::element::i32, ov::Shape{}, std::vector{0});
        auto select_node = std::make_shared<v1::Select>(read_value_spec_tokens, num_added_tokens_const, zero_constant);
        subtract_node->input(1).replace_source_output(select_node->output(0));
    }
    
    model->add_sinks({std::make_shared<v6::Assign>(max_length_rv, max_length_var)});
    model->add_variables({max_length_var});

    std::vector<std::shared_ptr<ov::Node>> ragged_to_dense_nodes;
    for (auto node: model->get_ordered_ops()) {
        if (strcmp(node->get_type_info().name, "RaggedToDense") == 0) {
            ragged_to_dense_nodes.emplace_back(node);
        }
    }

    if (ragged_to_dense_nodes.size() < 1) {
        return true;  // true since at this point we already have modified the graph.s
    }
    
    // By default do not pad to max_length
    auto pad_to_max_length_var = std::make_shared<op::util::Variable>(op::util::VariableInfo{ov::Shape{1}, ov::element::boolean, PAD_TO_MAX_LENGTH_VAR_ID});
    auto default_false_const = std::make_shared<v0::Constant>(ov::element::boolean, ov::Shape{1}, std::vector{false});
    auto pad_to_max_length_rv = std::make_shared<v6::ReadValue>(default_false_const, pad_to_max_length_var);
    model->add_sinks({std::make_shared<v6::Assign>(pad_to_max_length_rv, pad_to_max_length_var)});
    model->add_variables({pad_to_max_length_var});
    
    auto zero_constant = std::make_shared<v0::Constant>(ov::element::i32, ov::Shape{}, std::vector{0});
    auto select_node = std::make_shared<v1::Select>(pad_to_max_length_rv, max_length_rv, zero_constant);

    for (auto ragged_to_dense_node : ragged_to_dense_nodes) {
        if (!ragged_to_dense_node) {
            return true;  // true since at this point we already have modified the graph.s
        }
        
        auto max_op = std::make_shared<v1::Maximum>(ragged_to_dense_node->input_value(3), select_node);
        ragged_to_dense_node->input(3).replace_source_output(max_op->output(0));
    }

    return true;
}
