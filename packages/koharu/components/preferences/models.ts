import type {
  DetectionModel,
  InpaintingModel,
  OcrModel,
  PipelineConfig,
  Stage,
} from '@koharu/bridge/protocol'

export type PipelineModel = DetectionModel | OcrModel | InpaintingModel
export type ModelStage = Exclude<Stage, 'translation'>
export type ModelName = PipelineModel['model']

export const modelOptions = {
  detection: ['koharu-layout-rfdetr-seg-2xl', 'koharu-layout-highres'],
  ocr: ['paddleocr-vl-1.6', 'manga-ocr', 'manga-ocr-ar', 'baberu-ocr', 'hayai-ocr', 'dots.ocr'],
  inpainting: [
    'lama',
    'aot-inpainting',
    'flux2-klein',
    'flux3-klein',
    'minimax-h3-image',
    'rorem-mixed',
  ],
} satisfies Record<ModelStage, ModelName[]>

export const modelNames: Record<ModelName, string> = {
  'koharu-layout-rfdetr-seg-2xl': 'Koharu Layout RF-DETR Seg 2XL',
  'koharu-layout-highres': 'Koharu Layout HighRes',
  'paddleocr-vl-1.6': 'PaddleOCR-VL 1.6',
  'manga-ocr': 'Manga OCR',
  'manga-ocr-ar': 'Manga OCR AR',
  'baberu-ocr': 'Baberu OCR',
  'hayai-ocr': 'Hayai OCR',
  'dots.ocr': 'dots.ocr',
  lama: 'LaMa',
  'aot-inpainting': 'AOT Inpainting',
  'flux2-klein': 'FLUX.2 Klein',
  'flux3-klein': 'FLUX.3 Klein',
  'minimax-h3-image': 'MiniMax-H3-Image',
  'rorem-mixed': 'RORem Mixed',
}

export const FLUX3_KLEIN_PROMPT =
  "Remove only the text and lettering. Keep the speech bubble's flat fill colour, outline and tail exactly as they are. Fill the removed text with the surrounding bubble colour."

export const MINIMAX_H3_IMAGE_PROMPT =
  'Edit <Picture 1>. Remove all text and lettering, and reconstruct the hidden artwork and background in the same style.'

export function defaultModel(model: ModelName): PipelineModel {
  switch (model) {
    case 'koharu-layout-rfdetr-seg-2xl':
    case 'koharu-layout-highres':
      return { model, text_threshold: null, bubble_threshold: null, panel_threshold: null }
    case 'paddleocr-vl-1.6':
    case 'manga-ocr':
    case 'manga-ocr-ar':
    case 'baberu-ocr':
    case 'hayai-ocr':
    case 'dots.ocr':
    case 'lama':
    case 'aot-inpainting':
      return { model }
    case 'flux2-klein':
      return { model, prompt: 'Remove the text and reconstruct the background.' }
    case 'flux3-klein':
      return { model, prompt: FLUX3_KLEIN_PROMPT }
    case 'minimax-h3-image':
      return { model, prompt: MINIMAX_H3_IMAGE_PROMPT }
    case 'rorem-mixed':
      return { model }
  }
}

export function stageModel(config: PipelineConfig, stage: ModelStage): PipelineModel {
  const selected =
    stage === 'detection' ? config.detection : stage === 'ocr' ? config.ocr : config.inpainting
  if (selected) {
    const profile = config.processor?.[selected.model as keyof typeof config.processor]
    return profile ? ({ model: selected.model, ...profile } as PipelineModel) : selected
  }
  return defaultModel(modelOptions[stage][0]!)
}

export function replaceStage(
  config: PipelineConfig,
  stage: ModelStage,
  model: PipelineModel,
): PipelineConfig {
  switch (stage) {
    case 'detection': {
      const detection = model as DetectionModel
      return {
        ...config,
        detection,
        processor: {
          ...config.processor,
          [detection.model]: {
            text_threshold: detection.text_threshold ?? null,
            bubble_threshold: detection.bubble_threshold ?? null,
            panel_threshold: detection.panel_threshold ?? null,
          },
        },
      }
    }
    case 'ocr':
      return { ...config, ocr: model as OcrModel }
    case 'inpainting':
      return {
        ...config,
        inpainting: model as InpaintingModel,
        processor: {
          ...config.processor,
          ...(model.model === 'flux2-klein'
            ? { 'flux2-klein': { prompt: model.prompt ?? undefined } }
            : {}),
          ...(model.model === 'flux3-klein'
            ? { 'flux3-klein': { prompt: model.prompt ?? undefined } }
            : {}),
          ...(model.model === 'minimax-h3-image'
            ? { 'minimax-h3-image': { prompt: model.prompt ?? undefined } }
            : {}),
          ...(model.model === 'rorem-mixed'
            ? {
                'rorem-mixed': {
                  prompt: model.prompt ?? undefined,
                  negative_prompt: model.negative_prompt ?? undefined,
                },
              }
            : {}),
        },
      }
  }
}
