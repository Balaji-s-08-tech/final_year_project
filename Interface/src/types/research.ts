export type ResearchStatus = 'VERIFIED' | 'PENDING' | 'BLOCKED' | 'DEMO';

export type ViewMode = 'dashboard' | 'experiments' | 'pipeline' | 'models' | 'quantization' | 'benchmarks' | 'results' | 'system' | 'notes' | 'presentation';

export type StageExecutionStatus = 'idle' | 'pending' | 'running' | 'completed' | 'failed' | 'blocked';

export type ExecutionEnvironment = 'LOCAL' | 'CLOUD' | 'HYBRID';

export type QuantizationMethod = 'None' | 'LLM.int8' | 'AWQ INT4';

export type PruningCriterion = 'Taylor FO' | 'L1 Norm' | 'Hessian' | 'Wanda' | 'Random';

export interface ModelSpecification {
  id: string;
  name: string;
  family: string;
  totalParameters: string;
  visionEncoder: {
    name: string;
    architecture: string;
    parameters: string;
    resolution: string;
    patchSize: number;
    hiddenDimension: number;
  };
  projector: {
    type: string;
    layers: number;
    inputDim: number;
    outputDim: number;
  };
  languageBackbone: {
    name: string;
    architecture: string;
    parameters: string;
    layers: number;
    hiddenDimension: number;
    attentionHeads: number;
    intermediateDimension: number;
    vocabularySize: number;
  };
  contextLength: number;
  license: string;
}

export interface PruningConfig {
  method: string;
  targetRatio: number;
  mode: 'layer-wise' | 'width-wise' | 'block-wise';
  criterion: PruningCriterion;
  targetModules: string[];
}

export interface RecoveryConfig {
  enabled: boolean;
  trainingMethod: 'LoRA' | 'Full Fine-Tuning' | 'Adapter Only';
  epochs: number;
  learningRate: number;
  loraRank: number;
  loraAlpha: number;
  batchSize: number;
}

export interface AWQConfig {
  calibrationDataset: 'Pile-val' | 'ShareGPT4V' | 'COCO-QA' | 'Wikitext-2';
  calibrationSamples: number;
  weightBits: 4;
  groupSize: 64 | 128;
  symmetric: boolean;
  activationAwareSearch: boolean;
  dualityRatio: number;
}

export interface ExperimentConfig {
  id: string;
  name: string;
  model: string;
  pruning: PruningConfig;
  recovery: RecoveryConfig;
  quantization: {
    method: QuantizationMethod;
    awq?: AWQConfig;
  };
  execution: ExecutionEnvironment;
  createdAt: string;
}

export interface PipelineStageInfo {
  id: number;
  key: string;
  name: string;
  description: string;
  status: StageExecutionStatus;
  progress: number;
  detail: string;
  runtime?: string;
  logs?: string[];
  executionType: 'local' | 'cloud' | 'hybrid';
  researchStatus: ResearchStatus;
}

export interface LogEntry {
  id: string;
  timestamp: string;
  level: 'INFO' | 'WARNING' | 'ERROR' | 'SUCCESS';
  stage: string;
  message: string;
}

export interface MetricValue {
  name: string;
  value: string | number | null;
  unit?: string;
  status: ResearchStatus;
  note?: string;
}

export interface ModelComparisonRow {
  variant: 'Original' | 'Pruned (20%)' | 'Pruned + LLM.int8' | 'Pruned + AWQ INT4';
  parameters: MetricValue;
  checkpointSize: MetricValue;
  weightPrecision: MetricValue;
  vramTraining: MetricValue;
  vramInference: MetricValue;
  latencyPerToken: MetricValue;
  accuracyPOPE: MetricValue;
  accuracyScienceQA: MetricValue;
  f1Score: MetricValue;
  mcc: MetricValue;
  rocAuc: MetricValue;
  status: ResearchStatus;
}

export interface HardwareSpec {
  gpuName: string;
  vramTotalGb: number;
  vramAllocatedGb: number;
  systemRamTotalGb: number;
  systemRamAllocatedGb: number;
  driverVersion: string;
  cudaVersion: string;
  torchVersion: string;
  flashAttention: boolean;
  recommendedMode: ExecutionEnvironment;
}

export interface VerifiedDevelopmentItem {
  id: string;
  title: string;
  component: string;
  status: ResearchStatus;
  timestamp: string;
  detail: string;
  verificationEvidence: string;
}
