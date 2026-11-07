import {
  ModelSpecification,
  HardwareSpec,
  VerifiedDevelopmentItem,
  PipelineStageInfo,
  ModelComparisonRow,
  ExperimentConfig,
  LogEntry
} from '../types/research';

export const BUNNY_MODEL_SPEC: ModelSpecification = {
  id: 'bunny-v1_0-3b',
  name: 'Bunny-v1_0-3B',
  family: 'Bunny-LVLM',
  totalParameters: '~3.12B',
  visionEncoder: {
    name: 'google/siglip-so400m-patch14-384',
    architecture: 'SigLIP Vision Transformer',
    parameters: '428M',
    resolution: '384 × 384',
    patchSize: 14,
    hiddenDimension: 1152
  },
  projector: {
    type: '2-Layer MLP with GELU activation',
    layers: 2,
    inputDim: 1152,
    outputDim: 2560
  },
  languageBackbone: {
    name: 'microsoft/phi-2',
    architecture: 'Causal Transformer (Phi-2)',
    parameters: '2.78B',
    layers: 32,
    hiddenDimension: 2560,
    attentionHeads: 32,
    intermediateDimension: 10240,
    vocabularySize: 51200
  },
  contextLength: 2048,
  license: 'Apache-2.0 / Research Use'
};

export const HARDWARE_SPEC: HardwareSpec = {
  gpuName: 'NVIDIA GeForce RTX 3050 Laptop GPU',
  vramTotalGb: 6.0,
  vramAllocatedGb: 2.4,
  systemRamTotalGb: 16.0,
  systemRamAllocatedGb: 8.2,
  driverVersion: '535.183.01',
  cudaVersion: 'CUDA 12.1',
  torchVersion: '2.1.2+cu121',
  flashAttention: true,
  recommendedMode: 'HYBRID'
};

export const VERIFIED_STATUS_ITEMS: VerifiedDevelopmentItem[] = [
  {
    id: 'V-01',
    title: 'Bunny-v1_0-3B Environment Configuration',
    component: 'Environment / Conda',
    status: 'VERIFIED',
    timestamp: 'Phase 1 Milestone',
    detail: 'Torch, torchvision, transformers, timm, accelerate, and custom SigLIP image processor successfully linked in environment.',
    verificationEvidence: 'Environment activation check passed without missing symbol warnings.'
  },
  {
    id: 'V-02',
    title: 'Phi-2 Language Backbone Weight Ingestion',
    component: 'VLM/model_loader.py',
    status: 'VERIFIED',
    timestamp: 'Phase 1 Milestone',
    detail: 'Hugging Face weights for microsoft/phi-2 successfully loaded into CUDA memory; model head & embeddings verified.',
    verificationEvidence: 'Local load check verified layer tensors across 32 transformer blocks.'
  },
  {
    id: 'V-03',
    title: 'AWQ INT4 Core Integration',
    component: 'VLM/quantization/awq/',
    status: 'VERIFIED',
    timestamp: 'Phase 2 Milestone',
    detail: 'Activation-aware scale search algorithm and W4A16 kernel binding integrated into the VLM quantization module.',
    verificationEvidence: 'Module test imported successfully with W4A16 GEMM kernel registration.'
  },
  {
    id: 'V-04',
    title: 'AWQ Language-Only Smoke Test',
    component: 'VLM/quantization/awq/smoke_test.py',
    status: 'VERIFIED',
    timestamp: 'Phase 2 Milestone',
    detail: 'Completed test run on language backbone subset with synthetic calibration samples; verified layer replacement.',
    verificationEvidence: 'Replaced standard Linear layers with packed W4A16 AWQ Linear modules without tensor shape mismatch.'
  },
  {
    id: 'V-05',
    title: 'Packed AWQ Layer Replacement Validation',
    component: 'VLM/quantization/awq/awq_quantize.py',
    status: 'VERIFIED',
    timestamp: 'Phase 2 Milestone',
    detail: 'Replaced PyTorch nn.Linear modules with custom packed INT4 weights and scales (group_size=128, symmetric=True).',
    verificationEvidence: 'State dict parameter introspection confirmed int32 packed storage.'
  },
  {
    id: 'V-06',
    title: 'Structural Pruning Smoke Test & Checkpoint Export',
    component: 'LLM-Pruner/prune.py',
    status: 'VERIFIED',
    timestamp: 'Phase 2 Milestone',
    detail: 'Generated a structurally pruned checkpoint (width-wise channels removed according to Taylor first-order criterion).',
    verificationEvidence: 'Checkpoint successfully exported to disk with reduced intermediate matrix dimensions.'
  },
  {
    id: 'V-07',
    title: 'Full End-to-End Bunny Recovery Fine-Tuning',
    component: 'VLM/train/train.py',
    status: 'BLOCKED',
    timestamp: 'Local Limitation',
    detail: 'Full multimodal recovery with vision encoder backprop exceeds 6GB VRAM on local RTX 3050. Requires Cloud GPU allocation.',
    verificationEvidence: 'CUDA Out-Of-Memory encountered when batch_size=1 with gradients enabled on local machine.'
  },
  {
    id: 'V-08',
    title: 'Comprehensive Recovered Model Multimodal Benchmark',
    component: 'benchmark/eval_pope_mme.py',
    status: 'PENDING',
    timestamp: 'Awaiting Cloud Recovery',
    detail: 'Full evaluation across POPE, MME, ScienceQA, and TextVQA will execute once the recovered pruned checkpoint is produced.',
    verificationEvidence: 'Awaiting cloud checkpoint generation.'
  },
  {
    id: 'V-09',
    title: 'Final Downstream Accuracy Comparison (FP16 vs INT8 vs AWQ)',
    component: 'evaluation/accuracy_matrix.py',
    status: 'PENDING',
    timestamp: 'Awaiting Execution',
    detail: 'Direct comparative scientific evaluation between uncompressed, pruned, pruned+INT8, and pruned+AWQ INT4.',
    verificationEvidence: 'Marked pending in experimental matrix to prevent unverified claims.'
  },
  {
    id: 'V-10',
    title: 'Hardware Inference Latency & Energy Profiling',
    component: 'benchmark/latency_profiler.py',
    status: 'PENDING',
    timestamp: 'Awaiting Final Build',
    detail: 'Hardware time-to-first-token (TTFT) and throughput tokens/sec across different batch sizes on target hardware.',
    verificationEvidence: 'Execution scheduled following recovered quantized model generation.'
  }
];

export const INITIAL_PIPELINE_STAGES: PipelineStageInfo[] = [
  {
    id: 1,
    key: 'model_loading',
    name: 'Model Loading',
    description: 'Ingest SigLIP-SO400M vision encoder and Phi-2 language backbone into memory.',
    status: 'completed',
    progress: 100,
    detail: 'Weights loaded into memory; SigLIP (428M) + Projector + Phi-2 (2.78B).',
    runtime: '14.2s',
    executionType: 'local',
    researchStatus: 'VERIFIED'
  },
  {
    id: 2,
    key: 'calibration',
    name: 'Activation Calibration',
    description: 'Pass multimodal calibration samples through model to capture channel activation dynamics.',
    status: 'completed',
    progress: 100,
    detail: 'Processed 128 calibration sequences through forward hooks to collect activation variance.',
    runtime: '38.5s',
    executionType: 'local',
    researchStatus: 'VERIFIED'
  },
  {
    id: 3,
    key: 'importance_estimation',
    name: 'Importance Estimation',
    description: 'Calculate gradient-based Taylor first-order sensitivity for channel & layer structures.',
    status: 'completed',
    progress: 100,
    detail: 'Importance scores computed for attention query/key/value projections and MLP intermediate layers.',
    runtime: '26.8s',
    executionType: 'local',
    researchStatus: 'VERIFIED'
  },
  {
    id: 4,
    key: 'structural_pruning',
    name: 'Structural Pruning',
    description: 'Physically remove identified low-importance channels from language backbone matrices.',
    status: 'completed',
    progress: 100,
    detail: 'Pruned 20% width from intermediate dimensions; shapes resized from 10240 to 8192.',
    runtime: '11.4s',
    executionType: 'local',
    researchStatus: 'VERIFIED'
  },
  {
    id: 5,
    key: 'recovery',
    name: 'Recovery Fine-Tuning',
    description: 'Multimodal post-pruning parameter recovery via LoRA adapter optimization.',
    status: 'blocked',
    progress: 0,
    detail: 'Blocked on local hardware (RTX 3050 6GB). Exceeds local VRAM capacity; scheduled for Cloud GPU.',
    runtime: '--',
    executionType: 'cloud',
    researchStatus: 'BLOCKED'
  },
  {
    id: 6,
    key: 'awq_scale_search',
    name: 'AWQ Scale Search',
    description: 'Grid search per-channel protection scales that minimize quantization error on salient channels.',
    status: 'pending',
    progress: 0,
    detail: 'Grid search over alpha parameter in range [0, 1] on recovered checkpoint.',
    runtime: '--',
    executionType: 'hybrid',
    researchStatus: 'PENDING'
  },
  {
    id: 7,
    key: 'weight_quantization',
    name: 'AWQ INT4 Quantization',
    description: 'Apply optimal scales and pack weights into INT4 format with group size 128.',
    status: 'pending',
    progress: 0,
    detail: 'Linear transformation replaced by W4A16 packed kernel layers.',
    runtime: '--',
    executionType: 'hybrid',
    researchStatus: 'PENDING'
  },
  {
    id: 8,
    key: 'checkpoint_saving',
    name: 'Checkpoint Saving',
    description: 'Export compressed architecture and serialized packed weights to disk.',
    status: 'pending',
    progress: 0,
    detail: 'Serialized pruned+AWQ model format ready for deployment and evaluation.',
    runtime: '--',
    executionType: 'local',
    researchStatus: 'PENDING'
  },
  {
    id: 9,
    key: 'benchmarking',
    name: 'Benchmarking',
    description: 'Profile VRAM allocation, time to first token, and continuous token throughput.',
    status: 'pending',
    progress: 0,
    detail: 'Hardware profiling across local RTX 3050 and baseline cloud instances.',
    runtime: '--',
    executionType: 'hybrid',
    researchStatus: 'PENDING'
  },
  {
    id: 10,
    key: 'evaluation',
    name: 'Downstream Evaluation',
    description: 'Evaluate multimodal accuracy on standard benchmarks (POPE, MME, ScienceQA).',
    status: 'pending',
    progress: 0,
    detail: 'Awaiting recovered model output. No fabricated accuracy scores allowed.',
    runtime: '--',
    executionType: 'cloud',
    researchStatus: 'PENDING'
  }
];

export const INITIAL_EXPERIMENTS: ExperimentConfig[] = [
  {
    id: 'EXP-0001',
    name: 'Smoke Test: Phi-2 AWQ INT4 Only',
    model: 'Bunny-v1_0-3B (Backbone only)',
    pruning: {
      method: 'None',
      targetRatio: 0.0,
      mode: 'width-wise',
      criterion: 'Taylor FO',
      targetModules: []
    },
    recovery: {
      enabled: false,
      trainingMethod: 'LoRA',
      epochs: 0,
      learningRate: 0.0002,
      loraRank: 16,
      loraAlpha: 32,
      batchSize: 1
    },
    quantization: {
      method: 'AWQ INT4',
      awq: {
        calibrationDataset: 'Pile-val',
        calibrationSamples: 32,
        weightBits: 4,
        groupSize: 128,
        symmetric: true,
        activationAwareSearch: true,
        dualityRatio: 0.5
      }
    },
    execution: 'LOCAL',
    createdAt: '2026-03-01 14:20'
  },
  {
    id: 'EXP-0002',
    name: 'Smoke Test: 20% Width Pruning Checkpoint',
    model: 'Bunny-v1_0-3B',
    pruning: {
      method: 'LLM-Pruner Structured',
      targetRatio: 0.20,
      mode: 'width-wise',
      criterion: 'Taylor FO',
      targetModules: ['q_proj', 'k_proj', 'v_proj', 'fc1', 'fc2']
    },
    recovery: {
      enabled: false,
      trainingMethod: 'LoRA',
      epochs: 0,
      learningRate: 0.0002,
      loraRank: 16,
      loraAlpha: 32,
      batchSize: 1
    },
    quantization: {
      method: 'None'
    },
    execution: 'LOCAL',
    createdAt: '2026-03-03 10:15'
  },
  {
    id: 'EXP-0003',
    name: 'Cloud Target: 20% Pruned + LoRA Recovery + AWQ INT4',
    model: 'Bunny-v1_0-3B',
    pruning: {
      method: 'LLM-Pruner Structured',
      targetRatio: 0.20,
      mode: 'width-wise',
      criterion: 'Taylor FO',
      targetModules: ['q_proj', 'k_proj', 'v_proj', 'fc1', 'fc2']
    },
    recovery: {
      enabled: true,
      trainingMethod: 'LoRA',
      epochs: 3,
      learningRate: 0.0002,
      loraRank: 16,
      loraAlpha: 32,
      batchSize: 4
    },
    quantization: {
      method: 'AWQ INT4',
      awq: {
        calibrationDataset: 'ShareGPT4V',
        calibrationSamples: 128,
        weightBits: 4,
        groupSize: 128,
        symmetric: true,
        activationAwareSearch: true,
        dualityRatio: 0.5
      }
    },
    execution: 'CLOUD',
    createdAt: '2026-03-08 09:30'
  }
];

export const INITIAL_LOGS: LogEntry[] = [
  {
    id: 'log-1',
    timestamp: '14:20:01',
    level: 'INFO',
    stage: 'Environment',
    message: 'CUDA device identified: NVIDIA GeForce RTX 3050 Laptop GPU (Total VRAM: 6144 MB)'
  },
  {
    id: 'log-2',
    timestamp: '14:20:03',
    level: 'INFO',
    stage: 'Model Loading',
    message: 'Initializing Bunny-v1_0-3B architecture: SigLIP-SO400M + 2-layer MLP + Phi-2'
  },
  {
    id: 'log-3',
    timestamp: '14:20:12',
    level: 'INFO',
    stage: 'Model Loading',
    message: 'Phi-2 backbone loaded successfully: 32 transformer layers, 2560 hidden dimension'
  },
  {
    id: 'log-4',
    timestamp: '14:20:18',
    level: 'INFO',
    stage: 'Pruning Smoke Test',
    message: 'Initiating Taylor first-order sensitivity calculation on subset layers'
  },
  {
    id: 'log-5',
    timestamp: '14:20:25',
    level: 'SUCCESS',
    stage: 'Pruning Smoke Test',
    message: 'Structured pruning smoke test completed: intermediate dimension reduced 10240 -> 8192 (20% ratio)'
  },
  {
    id: 'log-6',
    timestamp: '14:20:31',
    level: 'INFO',
    stage: 'AWQ Integration',
    message: 'Validating AWQ W4A16 kernel binding: group_size=128, symmetric=True'
  },
  {
    id: 'log-7',
    timestamp: '14:20:44',
    level: 'SUCCESS',
    stage: 'AWQ Integration',
    message: 'Linear layers successfully replaced with packed AWQ INT4 layers in backbone smoke test'
  },
  {
    id: 'log-8',
    timestamp: '14:21:02',
    level: 'WARNING',
    stage: 'Recovery Constraint',
    message: 'Local execution warning: Full multimodal recovery requires >14GB VRAM. Local RTX 3050 limited to 6GB.'
  },
  {
    id: 'log-9',
    timestamp: '14:21:05',
    level: 'INFO',
    stage: 'System Router',
    message: 'Routing full recovery and downstream multimodal benchmark to Cloud GPU execution path.'
  }
];

export const MODEL_COMPARISON_DATA: ModelComparisonRow[] = [
  {
    variant: 'Original',
    parameters: { name: 'Parameters', value: '3.12B', status: 'VERIFIED', note: 'SigLIP 428M + Phi-2 2.78B' },
    checkpointSize: { name: 'Checkpoint', value: '6.24 GB', status: 'VERIFIED', note: 'FP16 precision' },
    weightPrecision: { name: 'Precision', value: 'FP16', status: 'VERIFIED' },
    vramTraining: { name: 'Training VRAM', value: '~18.4 GB', status: 'VERIFIED', note: 'Standard full multimodal' },
    vramInference: { name: 'Inference VRAM', value: '7.8 GB', status: 'VERIFIED', note: 'Baseline peak usage' },
    latencyPerToken: { name: 'Latency', value: 'Pending', status: 'PENDING', note: 'Profiling harness scheduled' },
    accuracyPOPE: { name: 'POPE', value: 'Pending', status: 'PENDING', note: 'Standard evaluation pending' },
    accuracyScienceQA: { name: 'ScienceQA', value: 'Pending', status: 'PENDING', note: 'Standard evaluation pending' },
    f1Score: { name: 'F1', value: 'Pending', status: 'PENDING' },
    mcc: { name: 'MCC', value: 'Pending', status: 'PENDING' },
    rocAuc: { name: 'ROC-AUC', value: 'Pending', status: 'PENDING' },
    status: 'VERIFIED'
  },
  {
    variant: 'Pruned (20%)',
    parameters: { name: 'Parameters', value: '~2.56B', status: 'VERIFIED', note: 'Structural width reduction' },
    checkpointSize: { name: 'Checkpoint', value: '5.12 GB', status: 'VERIFIED', note: 'FP16 pruned checkpoint generated' },
    weightPrecision: { name: 'Precision', value: 'FP16', status: 'VERIFIED' },
    vramTraining: { name: 'Training VRAM', value: '~14.2 GB', status: 'BLOCKED', note: 'Exceeds local 6GB' },
    vramInference: { name: 'Inference VRAM', value: 'Pending', status: 'PENDING', note: 'Measurement scheduled' },
    latencyPerToken: { name: 'Latency', value: 'Pending', status: 'PENDING', note: 'Awaiting post-recovery test' },
    accuracyPOPE: { name: 'POPE', value: 'Pending', status: 'PENDING', note: 'Awaiting post-recovery test' },
    accuracyScienceQA: { name: 'ScienceQA', value: 'Pending', status: 'PENDING', note: 'Awaiting post-recovery test' },
    f1Score: { name: 'F1', value: 'Pending', status: 'PENDING' },
    mcc: { name: 'MCC', value: 'Pending', status: 'PENDING' },
    rocAuc: { name: 'ROC-AUC', value: 'Pending', status: 'PENDING' },
    status: 'VERIFIED'
  },
  {
    variant: 'Pruned + LLM.int8',
    parameters: { name: 'Parameters', value: '~2.56B', status: 'VERIFIED', note: 'Identical pruned architecture' },
    checkpointSize: { name: 'Checkpoint', value: 'Pending', status: 'PENDING', note: 'INT8 baseline export pending' },
    weightPrecision: { name: 'Precision', value: 'INT8 (8-bit)', status: 'VERIFIED' },
    vramTraining: { name: 'Training VRAM', value: 'N/A (Post-train)', status: 'VERIFIED' },
    vramInference: { name: 'Inference VRAM', value: 'Pending', status: 'PENDING', note: 'Comparative run scheduled' },
    latencyPerToken: { name: 'Latency', value: 'Pending', status: 'PENDING', note: 'Comparative run scheduled' },
    accuracyPOPE: { name: 'POPE', value: 'Pending', status: 'PENDING', note: 'Comparative run scheduled' },
    accuracyScienceQA: { name: 'ScienceQA', value: 'Pending', status: 'PENDING', note: 'Comparative run scheduled' },
    f1Score: { name: 'F1', value: 'Pending', status: 'PENDING' },
    mcc: { name: 'MCC', value: 'Pending', status: 'PENDING' },
    rocAuc: { name: 'ROC-AUC', value: 'Pending', status: 'PENDING' },
    status: 'PENDING'
  },
  {
    variant: 'Pruned + AWQ INT4',
    parameters: { name: 'Parameters', value: '~2.56B', status: 'VERIFIED', note: 'Identical pruned architecture' },
    checkpointSize: { name: 'Checkpoint', value: 'Pending', status: 'PENDING', note: 'INT4 packed export pending' },
    weightPrecision: { name: 'Precision', value: 'INT4 (4-bit)', status: 'VERIFIED' },
    vramTraining: { name: 'Training VRAM', value: 'N/A (Post-train)', status: 'VERIFIED' },
    vramInference: { name: 'Inference VRAM', value: 'Pending', status: 'PENDING', note: 'Comparative run scheduled' },
    latencyPerToken: { name: 'Latency', value: 'Pending', status: 'PENDING', note: 'Comparative run scheduled' },
    accuracyPOPE: { name: 'POPE', value: 'Pending', status: 'PENDING', note: 'Comparative run scheduled' },
    accuracyScienceQA: { name: 'ScienceQA', value: 'Pending', status: 'PENDING', note: 'Comparative run scheduled' },
    f1Score: { name: 'F1', value: 'Pending', status: 'PENDING' },
    mcc: { name: 'MCC', value: 'Pending', status: 'PENDING' },
    rocAuc: { name: 'ROC-AUC', value: 'Pending', status: 'PENDING' },
    status: 'PENDING'
  }
];

export const PROJECT_PRESENTATION_DATA = {
  title: 'Enhancing Structured Pruning with Activation-Aware Weight Quantization (AWQ) for Efficient Large Vision-Language Models',
  researchProblem: 'Large Vision-Language Models (LVLMs) such as Bunny-3B exhibit severe memory and compute footprints that hinder edge deployment. While structured pruning physically reduces parameter dimensions and AWQ enables 4-bit weight compression, their compound interaction in multimodal architectures remains unexplored.',
  existingApproach: 'Current literature treats structured pruning (e.g., LLM-Pruner) and post-training quantization as disjoint optimization steps, or relies on 8-bit quantization (LLM.int8) which still incurs substantial memory bandwidth pressure on edge accelerators.',
  proposedImprovement: 'A unified compression pipeline: (1) Taylor-criterion structured pruning on the language backbone, (2) LoRA-based multimodal recovery, and (3) Activation-Aware Weight Quantization (AWQ) INT4 with calibration tuned specifically on pruned activation dynamics.',
  hardwareConstraint: 'Local experiments are executed on an NVIDIA GeForce RTX 3050 Laptop GPU (6 GB VRAM, 16 GB RAM). Full multimodal recovery fine-tuning requires cloud GPU scaling (>14 GB VRAM), establishing a Hybrid local-cloud execution strategy.',
  expectedOutcome: 'Characterization of whether structured pruning reshapes the activation outlier landscape, and identification of the critical pruning ratio threshold where AWQ INT4 retains quality advantages over INT8 baselines.',
  researchQuestions: [
    {
      q: 'Does structural pruning change the activation-aware importance landscape used by AWQ?',
      context: 'When channels are removed via Taylor sensitivity, do remaining channel activation distributions develop new outliers that require recalibrated AWQ protection scales?'
    },
    {
      q: 'At what pruning ratio does AWQ retain an advantage over the INT8 baseline?',
      context: 'Investigating if compounding high pruning ratios (>30%) with 4-bit quantization triggers cumulative degradation that surpasses the threshold of recovery.'
    }
  ]
};

export const REPOSITORY_MODULE_STRUCTURE = [
  {
    name: 'LLM-Pruner/',
    description: 'Structured pruning engine containing dependency graphs and Taylor sensitivity analyzers.',
    files: [
      { name: 'prune.py', role: 'Main entry point for structural channel and block pruning.' },
      { name: 'dependency.py', role: 'Inter-layer dependency graph builder to preserve tensor compatibility.' },
      { name: 'importance.py', role: 'First-order Taylor expansion importance score calculation.' }
    ]
  },
  {
    name: 'VLM/',
    description: 'Core vision-language model architecture wrapping SigLIP, MLP projector, and Phi-2.',
    files: [
      { name: 'model_loader.py', role: 'Instantiates Bunny-v1_0-3B with multimodal projector bindings.' },
      { name: 'eval_pope.py', role: 'POPE object hallucination evaluation script for VLMs.' },
      { name: 'eval_scienceqa.py', role: 'Multimodal multiple-choice QA evaluation harness.' }
    ]
  },
  {
    name: 'VLM/quantization/awq/',
    description: 'Activation-Aware Weight Quantization implementation specialized for compressed VLMs.',
    files: [
      { name: 'awq_model_loader.py', role: 'Resolves pruned module architecture before quantization.' },
      { name: 'awq_calibration.py', role: 'Collects forward activation statistics across multimodal prompts.' },
      { name: 'awq_quantize.py', role: 'Performs grid search for per-channel scales & packs INT4 weights.' },
      { name: 'awq_config.py', role: 'Defines bit-width (4), group size (128), and search hyper-parameters.' },
      { name: 'benchmark.py', role: 'Profiles memory consumption and token generation latency.' }
    ]
  }
];
