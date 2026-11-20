'use client'

import { FolderInput, Plus, Trash2 } from 'lucide-react'
import { useState } from 'react'
import { useTranslation } from 'react-i18next'

import { call, dispatch } from '@/lib/backend'
import { useProject } from '@/lib/queries'
import { setGlossary } from '@/lib/scene'
import { commands, type Term } from '@koharu/ipc/protocol'
import { Button } from '@koharu/ui/components/button'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@koharu/ui/components/dropdown-menu'
import { Input } from '@koharu/ui/components/input'
import { ScrollArea } from '@koharu/ui/components/scroll-area'

type Row = { key: number; source: string; target: string; note: string }

const EMPTY: Term[] = []
let nextKey = 0
const rowOf = (term?: Term): Row => ({
  key: nextKey++,
  source: term?.source ?? '',
  target: term?.target ?? '',
  note: term?.note ?? '',
})

/** The rows that hold a term, a blank one left out. */
const filled = (rows: Row[]) => rows.filter((row) => row.source.trim() || row.target.trim())

const termsOf = (rows: Row[]): Term[] =>
  filled(rows).map(({ source, target, note }) => ({
    source: source.trim(),
    target: target.trim(),
    note: note.trim() || null,
  }))

const sameTerms = (left: Term[], right: Term[]) =>
  left.length === right.length &&
  left.every(
    (term, at) =>
      term.source === right[at]!.source &&
      term.target === right[at]!.target &&
      (term.note ?? null) === (right[at]!.note ?? null),
  )

/** The rows missing a source or a target, and the rows sharing a source, by key. */
function problems(rows: Row[]) {
  const kept = filled(rows)
  const incomplete = new Set(
    kept.filter((row) => !row.source.trim() || !row.target.trim()).map((row) => row.key),
  )
  const first = new Map<string, number>()
  const duplicate = new Set<number>()
  for (const row of kept) {
    const source = row.source.trim()
    if (!source) continue
    const seen = first.get(source)
    if (seen === undefined) first.set(source, row.key)
    else duplicate.add(seen).add(row.key)
  }
  return { incomplete, duplicate }
}

/** The project's glossary: the names, places, and terms translation renders one way, each a
 * source, its target, and a note telling it apart. Rows are a draft the glossary takes whole, as
 * one undo step, when a field is left or Enter pressed, a row removed, or terms imported, once
 * every row has both a source and a target and no source repeats; a blank row is dropped. The
 * glossary changed elsewhere, by Undo or the agent, replaces the draft. */
export function GlossaryPanel() {
  const { t } = useTranslation()
  const project = useProject().data
  const glossary = project?.glossary ?? EMPTY
  const [rows, setRows] = useState<Row[]>(() => glossary.map(rowOf))
  const [seen, setSeen] = useState(glossary)
  const [blocked, setBlocked] = useState(false)
  const [added, setAdded] = useState<number | null>(null)
  const [projects, setProjects] = useState<string[] | null>(null)
  const [importing, setImporting] = useState(false)

  if (!sameTerms(glossary, seen)) {
    setSeen(glossary)
    if (!sameTerms(termsOf(rows), glossary)) setRows(glossary.map(rowOf))
  }

  const { incomplete, duplicate } = problems(rows)
  const error = incomplete.size
    ? t('glossary.incomplete')
    : duplicate.size
      ? t('glossary.duplicate')
      : null

  const commit = (next: Row[]) => {
    const { incomplete, duplicate } = problems(next)
    const invalid = incomplete.size > 0 || duplicate.size > 0
    setBlocked(invalid)
    const terms = termsOf(next)
    if (!invalid && !sameTerms(terms, glossary)) dispatch(setGlossary, terms)
  }

  const update = (key: number, field: 'source' | 'target' | 'note', value: string) =>
    setRows((current) => current.map((row) => (row.key === key ? { ...row, [field]: value } : row)))

  const add = () => {
    const row = rowOf()
    setAdded(row.key)
    setRows((current) => [...current, row])
  }

  const remove = (key: number) => {
    const next = rows.filter((row) => row.key !== key)
    setRows(next)
    commit(next)
  }

  const listProjects = (open: boolean) => {
    if (!open) return
    void call(commands.listProjects)
      .then((names) => setProjects(names.filter((name) => name !== project?.name)))
      .catch(() => setProjects([]))
  }

  // An imported term whose source is listed already keeps this project's target.
  const importFrom = (name: string) => {
    setImporting(true)
    void call(commands.getProjectGlossary, name)
      .then((terms) => {
        const kept = filled(rows)
        const sources = new Set(kept.map((row) => row.source.trim()))
        const next = [...kept, ...terms.filter((term) => !sources.has(term.source)).map(rowOf)]
        setRows(next)
        commit(next)
      })
      .catch(() => undefined)
      .finally(() => setImporting(false))
  }

  const field = 'h-6 rounded-md bg-background px-1.5 text-[11px] md:text-[11px]'
  // Source, translation, and note side by side under their headings, as the branch's dialog laid
  // them out, then the remove button.
  const columns =
    'grid grid-cols-[minmax(0,1fr)_minmax(0,1fr)_minmax(0,1.2fr)_1.25rem] items-center gap-1'
  return (
    <div className='flex min-h-0 flex-1 flex-col'>
      <div className='flex h-8 shrink-0 items-center gap-1 border-b border-border/80 px-1.5'>
        <Button
          type='button'
          variant='ghost'
          size='xs'
          disabled={!project}
          className='h-6 gap-1 px-1.5 text-[10px]'
          onClick={add}
        >
          <Plus className='size-3' />
          {t('glossary.add')}
        </Button>
        <DropdownMenu onOpenChange={listProjects}>
          <DropdownMenuTrigger
            render={
              <Button
                type='button'
                variant='ghost'
                size='xs'
                disabled={!project || importing}
                aria-busy={importing}
                className='ml-auto h-6 gap-1 px-1.5 text-[10px]'
              />
            }
          >
            <FolderInput className='size-3' />
            {t('glossary.import')}
          </DropdownMenuTrigger>
          <DropdownMenuContent align='end' className='w-auto max-w-64'>
            {projects?.length === 0 && (
              <DropdownMenuItem disabled className='text-[11px]'>
                {t('glossary.noProjects')}
              </DropdownMenuItem>
            )}
            {projects?.map((name) => (
              <DropdownMenuItem key={name} className='text-[11px]' onClick={() => importFrom(name)}>
                <span className='truncate'>{name}</span>
              </DropdownMenuItem>
            ))}
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
      {rows.length === 0 ? (
        <p className='px-4 py-8 text-center text-[10px] leading-4 text-muted-foreground'>
          {t('glossary.description')}
        </p>
      ) : (
        <>
          <div
            aria-hidden
            className={`${columns} shrink-0 px-2 pt-1.5 text-[10px] text-muted-foreground`}
          >
            <span>{t('glossary.source')}</span>
            <span>{t('glossary.target')}</span>
            <span>{t('glossary.note')}</span>
          </div>
          <ScrollArea className='min-h-0 flex-1' viewportClassName='p-1'>
            <ul className='grid gap-1' aria-label={t('glossary.title')}>
              {rows.map((row) => (
                <li
                  key={row.key}
                  className={`${columns} group rounded-md p-1 focus-within:bg-accent hover:bg-accent/60`}
                  onBlur={(event) => {
                    if (!event.currentTarget.contains(event.relatedTarget)) commit(rows)
                  }}
                  onKeyDown={(event) => {
                    if (event.key === 'Enter' && !event.nativeEvent.isComposing) commit(rows)
                  }}
                >
                  <Input
                    value={row.source}
                    autoFocus={row.key === added}
                    aria-label={t('glossary.source')}
                    aria-invalid={
                      (blocked && (incomplete.has(row.key) || duplicate.has(row.key))) || undefined
                    }
                    placeholder={t('glossary.sourcePlaceholder')}
                    className={field}
                    onChange={(event) => update(row.key, 'source', event.currentTarget.value)}
                  />
                  <Input
                    value={row.target}
                    aria-label={t('glossary.target')}
                    aria-invalid={(blocked && incomplete.has(row.key)) || undefined}
                    placeholder={t('glossary.targetPlaceholder')}
                    className={field}
                    onChange={(event) => update(row.key, 'target', event.currentTarget.value)}
                  />
                  <Input
                    value={row.note}
                    aria-label={t('glossary.note')}
                    placeholder={t('glossary.notePlaceholder')}
                    className={field}
                    onChange={(event) => update(row.key, 'note', event.currentTarget.value)}
                  />
                  <Button
                    type='button'
                    variant='ghost'
                    size='icon-xs'
                    aria-label={t('glossary.remove')}
                    className='size-5 text-muted-foreground hover:text-destructive'
                    onClick={() => remove(row.key)}
                  >
                    <Trash2 className='size-3' />
                  </Button>
                </li>
              ))}
            </ul>
          </ScrollArea>
        </>
      )}
      {blocked && error && (
        <p
          role='alert'
          className='shrink-0 border-t border-border/80 px-2 py-1.5 text-[10px] text-destructive'
        >
          {error}
        </p>
      )}
    </div>
  )
}
