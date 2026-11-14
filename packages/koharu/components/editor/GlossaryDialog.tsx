'use client'

import { FolderInput, Plus, Trash2 } from 'lucide-react'
import { useState } from 'react'
import { useTranslation } from 'react-i18next'

import { call } from '@/lib/backend'
import { projectKey, refresh } from '@/lib/queries'
import { commands, type GlossaryEntry, type ProjectSummary } from '@koharu/bridge/protocol'
import { Button } from '@koharu/ui/components/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@koharu/ui/components/dialog'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@koharu/ui/components/dropdown-menu'
import { Input } from '@koharu/ui/components/input'

type Row = { key: number; source: string; target: string; note: string }

let nextRowKey = 0
const row = (entry?: GlossaryEntry): Row => ({
  key: nextRowKey++,
  source: entry?.source ?? '',
  target: entry?.target ?? '',
  note: entry?.note ?? '',
})

export function GlossaryDialog({
  open,
  project,
  glossary,
  onOpenChange,
}: {
  open: boolean
  project: string
  glossary: GlossaryEntry[]
  onOpenChange: (open: boolean) => void
}) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      {open && (
        <GlossaryEditor project={project} glossary={glossary} onClose={() => onOpenChange(false)} />
      )}
    </Dialog>
  )
}

function GlossaryEditor({
  project,
  glossary,
  onClose,
}: {
  project: string
  glossary: GlossaryEntry[]
  onClose: () => void
}) {
  const { t } = useTranslation()
  const [rows, setRows] = useState<Row[]>(() => (glossary.length > 0 ? glossary.map(row) : [row()]))
  const [saving, setSaving] = useState(false)
  const [projects, setProjects] = useState<ProjectSummary[] | null>(null)
  const [importing, setImporting] = useState(false)

  const filled = rows.filter((entry) => entry.source.trim() || entry.target.trim())
  const incomplete = new Set(
    filled.filter((entry) => !entry.source.trim() || !entry.target.trim()).map(({ key }) => key),
  )
  const seen = new Map<string, number>()
  const duplicates = new Set<number>()
  for (const entry of filled) {
    const source = entry.source.trim()
    if (!source) continue
    const first = seen.get(source)
    if (first === undefined) seen.set(source, entry.key)
    else duplicates.add(first).add(entry.key)
  }
  const error =
    incomplete.size > 0
      ? t('glossary.incomplete')
      : duplicates.size > 0
        ? t('glossary.duplicate')
        : null

  const update = (key: number, field: 'source' | 'target' | 'note', value: string) =>
    setRows((current) =>
      current.map((entry) => (entry.key === key ? { ...entry, [field]: value } : entry)),
    )

  const loadProjects = (open: boolean) => {
    if (!open) return
    void call(commands.listProjects)
      .then((list) => setProjects(list.filter(({ name }) => name !== project)))
      .catch(() => setProjects([]))
  }

  // Imported terms are staged as rows like any other edit; terms whose source is
  // already listed keep the current translation.
  const importFrom = (name: string) => {
    setImporting(true)
    void call(commands.getProjectGlossary, name)
      .then((entries) =>
        setRows((current) => {
          const kept = current.filter((entry) => entry.source.trim() || entry.target.trim())
          const sources = new Set(kept.map((entry) => entry.source.trim()))
          const added = entries.filter((entry) => !sources.has(entry.source)).map(row)
          const merged = [...kept, ...added]
          return merged.length > 0 ? merged : [row()]
        }),
      )
      .catch(() => undefined)
      .finally(() => setImporting(false))
  }

  const save = () => {
    if (error || saving) return
    setSaving(true)
    const entries = filled.map(({ source, target, note }) => ({
      source,
      target,
      note: note.trim() || null,
    }))
    void call(commands.setGlossary, entries)
      .then(() => refresh(projectKey))
      .then(onClose)
      .catch(() => undefined)
      .finally(() => setSaving(false))
  }

  return (
    <DialogContent className='flex max-h-[min(36rem,calc(100vh-4rem))] flex-col gap-3 p-4 sm:max-w-2xl'>
      <DialogHeader className='gap-1'>
        <DialogTitle className='text-[13px]'>{t('glossary.title')}</DialogTitle>
        <DialogDescription className='text-[11px]'>{t('glossary.description')}</DialogDescription>
      </DialogHeader>

      <div className='grid grid-cols-[minmax(0,1fr)_minmax(0,1fr)_minmax(0,1.2fr)_1.75rem] gap-x-1.5 px-0.5 text-[10px] text-muted-foreground'>
        <span>{t('glossary.source')}</span>
        <span>{t('glossary.target')}</span>
        <span>{t('glossary.note')}</span>
      </div>
      <div className='-mt-1.5 grid min-h-0 flex-1 auto-rows-min gap-1.5 overflow-y-auto p-0.5'>
        {rows.map((entry) => {
          const invalid = duplicates.has(entry.key) || incomplete.has(entry.key)
          return (
            <div
              key={entry.key}
              className='grid grid-cols-[minmax(0,1fr)_minmax(0,1fr)_minmax(0,1.2fr)_1.75rem] items-center gap-x-1.5'
            >
              <Input
                value={entry.source}
                aria-label={t('glossary.source')}
                aria-invalid={invalid || undefined}
                placeholder={t('glossary.sourcePlaceholder')}
                className='h-7 text-[11px]'
                onChange={(event) => update(entry.key, 'source', event.currentTarget.value)}
              />
              <Input
                value={entry.target}
                aria-label={t('glossary.target')}
                aria-invalid={incomplete.has(entry.key) || undefined}
                placeholder={t('glossary.targetPlaceholder')}
                className='h-7 text-[11px]'
                onChange={(event) => update(entry.key, 'target', event.currentTarget.value)}
              />
              <Input
                value={entry.note}
                aria-label={t('glossary.note')}
                placeholder={t('glossary.notePlaceholder')}
                className='h-7 text-[11px]'
                onChange={(event) => update(entry.key, 'note', event.currentTarget.value)}
              />
              <Button
                type='button'
                variant='ghost'
                size='icon-xs'
                aria-label={t('glossary.remove')}
                className='text-muted-foreground hover:text-destructive'
                onClick={() =>
                  setRows((current) => current.filter((candidate) => candidate.key !== entry.key))
                }
              >
                <Trash2 className='size-3.5' />
              </Button>
            </div>
          )
        })}
        <Button
          type='button'
          variant='ghost'
          size='sm'
          className='h-7 justify-self-start text-[11px] text-muted-foreground'
          onClick={() => setRows((current) => [...current, row()])}
        >
          <Plus className='size-3.5' />
          {t('glossary.add')}
        </Button>
      </div>

      <DialogFooter className='-mx-4 -mb-4 items-center p-3'>
        <div className='mr-auto flex min-w-0 items-center gap-2'>
          <DropdownMenu onOpenChange={loadProjects}>
            <DropdownMenuTrigger
              render={
                <Button
                  type='button'
                  variant='outline'
                  size='sm'
                  disabled={saving || importing}
                  aria-busy={importing}
                />
              }
            >
              <FolderInput className='size-3.5' />
              {t('glossary.import')}
            </DropdownMenuTrigger>
            <DropdownMenuContent align='start' className='w-auto max-w-72'>
              {projects?.length === 0 && (
                <DropdownMenuItem disabled className='text-[11px]'>
                  {t('glossary.noProjects')}
                </DropdownMenuItem>
              )}
              {projects?.map(({ name }) => (
                <DropdownMenuItem
                  key={name}
                  className='text-[11px]'
                  onClick={() => importFrom(name)}
                >
                  <span className='truncate'>{name}</span>
                </DropdownMenuItem>
              ))}
            </DropdownMenuContent>
          </DropdownMenu>
          {error && <p className='text-[11px] text-destructive'>{error}</p>}
        </div>
        <Button type='button' variant='outline' size='sm' disabled={saving} onClick={onClose}>
          {t('common.cancel')}
        </Button>
        <Button type='button' size='sm' disabled={Boolean(error) || saving} onClick={save}>
          {t('common.save')}
        </Button>
      </DialogFooter>
    </DialogContent>
  )
}
