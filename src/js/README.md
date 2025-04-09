# OpenVINO™ GenAI Node.js bindings (preview)

Use OpenVINO GenAI to deploy most popular Generative AI model pipelines that run on top of highly performant OpenVINO Runtime.

## DISCLAIMER

This is preview version, do not use it in production!
In this version, only the LLM pipeline is implemented for NodeJS.

## Quick Start

Install the **genai-node** package:
```bash
npm install genai-node
```

Use the **genai-node** package:
```js
import { LLMPipeline } from 'genai-node';

const pipe = await LLMPipeline(MODEL_PATH, device);

const input = 'What is the meaning of life?';
const config = { 'max_new_tokens': 100 };

await pipe.startChat();
const result = await pipe.generate(input, config, streamer);
await pipe.finishChat();

// Output all generation result
console.log(result);

function streamer(subword) {
  process.stdout.write(subword);
}
```

## Supported Platforms

- Windows x86
- Linux x86/ARM
- MacOS x86/ARM

## Build From Sources

Build OpenVINO™ GenAI JavaScript Bindings from sources following the [instructions](https://github.com/openvinotoolkit/openvino.genai/blob/master/src/js/BUILD.md#build-openvino-genai-nodejs-bindings-preview).

## License

The OpenVINO™ GenAI repository is licensed under [Apache License Version 2.0](LICENSE).
By contributing to the project, you agree to the license and copyright terms therein and release
your contribution under these terms.