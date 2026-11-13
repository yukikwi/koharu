'use client'

import {
  Brush,
  Eraser,
  Hand,
  Minus,
  MousePointer2,
  Pipette,
  Plus,
  ScanText,
  Sparkles,
  Type,
} from 'lucide-react'
import { useEffect, useState, type ReactElement, type ReactNode } from 'react'
import { useTranslation } from 'react-i18next'

import { ColorWell } from '@/components/controls/ColorWell'
import {
  defaultModel,
  modelNames,
  modelOptions,
  type ModelName,
} from '@/components/preferences/models'
import { savePreferences } from '@/lib/backend'
import { usePage } from '@/lib/queries'
import {
  isBrushTool,
  maxBrushDiameter,
  MIN_BRUSH_DIAMETER,
  receivePreferences,
  useKoharuStore,
  type CanvasTool,
} from '@/lib/store'
import { Button } from '@koharu/ui/components/button'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuGroup,
  DropdownMenuLabel,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuTrigger,
} from '@koharu/ui/components/dropdown-menu'
import {
  NumberField,
  NumberFieldDecrement,
  NumberFieldGroup,
  NumberFieldIncrement,
  NumberFieldInput,
} from '@koharu/ui/components/number-field'
import {
  Popover,
  PopoverContent,
  PopoverTitle,
  PopoverTrigger,
} from '@koharu/ui/components/popover'
import { Slider } from '@koharu/ui/components/slider'
import { Tooltip, TooltipContent, TooltipTrigger } from '@koharu/ui/components/tooltip'

const tools = [
  ['select', MousePointer2],
  ['text', Type],
  ['text_region', ScanText],
  ['draw', Brush],
  ['eraser', Eraser],
  ['color_picker', Pipette],
  ['remove', Sparkles],
  ['pan', Hand],
] as const satisfies ReadonlyArray<readonly [CanvasTool, typeof MousePointer2]>

export function ToolBar() {
  const { t } = useTranslation()
  const page = usePage().data
  const active = useKoharuStore((state) => state.tool)
  const brush = useKoharuStore((state) => state.brush)
  const setTool = useKoharuStore((state) => state.setTool)
  const setBrush = useKoharuStore((state) => state.setBrush)
  const shortcuts = useKoharuStore((state) => state.shortcuts)
  const hasBrush = isBrushTool(active)

  return (
    <aside className='absolute top-3 left-3 z-20 flex w-11 flex-col rounded-2xl border border-border bg-[var(--surface-floating)] p-1 shadow-[var(--shadow-toolrail)]'>
      <div className='flex flex-col items-center py-0.5'>
        {tools.map(([tool, Icon], index) => {
          const button = (
            <Button
              type='button'
              variant='ghost'
              size='icon'
              disabled={!page}
              aria-label={t(`tools.${tool}`)}
              data-active={active === tool}
              className='relative text-muted-foreground hover:bg-foreground/[0.06] hover:text-foreground disabled:opacity-30 data-[active=true]:bg-accent data-[active=true]:text-accent-foreground'
              onClick={() => setTool(tool)}
            />
          )
          const tooltip = (
            <TooltipContent side='right'>
              {t(`tools.${tool}`)}
              <span className='ml-2 opacity-60'>{shortcuts[tool].toUpperCase()}</span>
            </TooltipContent>
          )
          return (
            <div key={tool} className='contents'>
              {index === tools.length - 1 && <span className='my-1 h-px w-5 bg-border' />}
              {tool === 'remove' ? (
                <RemoveModelMenu button={button} tooltip={tooltip}>
                  <Icon className='size-4' />
                </RemoveModelMenu>
              ) : (
                <Tooltip>
                  <TooltipTrigger render={button}>
                    <Icon className='size-4' />
                  </TooltipTrigger>
                  {tooltip}
                </Tooltip>
              )}
            </div>
          )
        })}
      </div>

      {hasBrush && (
        <div className='flex flex-col items-center gap-1 border-t border-border/80 py-1.5'>
          {active === 'draw' && (
            <ColorWell value={brush.color} onChange={(color) => setBrush({ ...brush, color })} />
          )}
          <BrushSize
            value={brush.diameter}
            onChange={(diameter) => setBrush({ ...brush, diameter })}
          />
        </div>
      )}
    </aside>
  )
}

const REMOVE_MENU_HIDE_DELAY_MS = 3000

/// Clicking Remove also offers its model. It is the Remove brush's own model and
/// never changes the inpainting stage. The menu is non-modal so painting can
/// start at once, and it hides after the pointer stays away.
function RemoveModelMenu({
  button,
  tooltip,
  children,
}: {
  button: ReactElement
  tooltip: ReactNode
  children: ReactNode
}) {
  const { t } = useTranslation()
  const preferences = useKoharuStore((state) => state.preferences)
  const [open, setOpen] = useState(false)
  const [hovered, setHovered] = useState(false)
  const selected = preferences?.pipeline.remove.model ?? null

  useEffect(() => {
    if (!open || hovered) return
    const timer = window.setTimeout(() => setOpen(false), REMOVE_MENU_HIDE_DELAY_MS)
    return () => window.clearTimeout(timer)
  }, [hovered, open])

  const select = (name: ModelName) => {
    setOpen(false)
    if (!preferences || name === selected) return
    // Only the model is chosen here; the backend applies its saved prompt profile.
    const remove = defaultModel(name) as typeof preferences.pipeline.remove
    void savePreferences(
      { ...preferences.pipeline, remove },
      preferences.providers,
      preferences.typesetting,
    )
      .then(receivePreferences)
      .catch(() => undefined)
  }

  const hover = {
    onPointerEnter: () => setHovered(true),
    onPointerLeave: () => setHovered(false),
  }

  return (
    <DropdownMenu modal={false} open={open} onOpenChange={setOpen}>
      <Tooltip>
        <DropdownMenuTrigger render={<TooltipTrigger render={button} />} {...hover}>
          {children}
        </DropdownMenuTrigger>
        {tooltip}
      </Tooltip>
      <DropdownMenuContent side='right' align='start' sideOffset={8} className='w-44' {...hover}>
        <DropdownMenuGroup>
          <DropdownMenuLabel className='text-[11px]'>{t('tools.removeModel')}</DropdownMenuLabel>
          <DropdownMenuRadioGroup
            value={selected}
            onValueChange={(name) => select(name as ModelName)}
          >
            {modelOptions.inpainting.map((name) => (
              <DropdownMenuRadioItem key={name} value={name} className='text-[11px]'>
                {modelNames[name]}
              </DropdownMenuRadioItem>
            ))}
          </DropdownMenuRadioGroup>
        </DropdownMenuGroup>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

function BrushSize({ value, onChange }: { value: number; onChange: (value: number) => void }) {
  const { t } = useTranslation()
  const roundedValue = Math.round(value)
  const page = usePage().data
  const maxSize = maxBrushDiameter(page?.size ?? { width: 2048, height: 2048 })

  return (
    <Popover>
      <Tooltip>
        <PopoverTrigger
          render={
            <TooltipTrigger
              render={
                <Button
                  type='button'
                  variant='ghost'
                  size='icon'
                  aria-label={t('tools.brushSizePixels', { size: roundedValue })}
                  className='size-8 rounded-xl font-sans text-[11px] leading-none font-medium tracking-[-0.02em] text-muted-foreground tabular-nums hover:bg-foreground/[0.06] hover:text-foreground'
                />
              }
            >
              {roundedValue}
            </TooltipTrigger>
          }
        />
        <TooltipContent side='right'>
          {t('tools.brushSizeShort', { size: roundedValue })}
        </TooltipContent>
      </Tooltip>
      <PopoverContent
        side='right'
        align='center'
        sideOffset={8}
        className='w-48 gap-2.5 rounded-xl p-2.5'
      >
        <div className='flex items-center justify-between gap-3'>
          <PopoverTitle className='text-[11px]'>{t('tools.brushSize')}</PopoverTitle>
          <NumberField
            min={MIN_BRUSH_DIAMETER}
            step={1}
            value={roundedValue}
            className='w-20'
            onValueChange={(next) => {
              if (next !== null) onChange(Math.max(next, MIN_BRUSH_DIAMETER))
            }}
          >
            <NumberFieldGroup className='h-7'>
              <NumberFieldDecrement aria-label={t('tools.decreaseBrushSize')}>
                <Minus />
              </NumberFieldDecrement>
              <NumberFieldInput aria-label={t('tools.brushSizeInput')} />
              <NumberFieldIncrement aria-label={t('tools.increaseBrushSize')}>
                <Plus />
              </NumberFieldIncrement>
            </NumberFieldGroup>
          </NumberField>
        </div>
        <Slider
          aria-label={t('tools.brushSize')}
          min={MIN_BRUSH_DIAMETER}
          max={maxSize}
          step={1}
          value={value}
          className='py-1 [&_[data-slot=slider-thumb]]:size-2.5'
          onValueChange={onChange}
        />
      </PopoverContent>
    </Popover>
  )
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, Number.isFinite(value) ? value : min))
}
