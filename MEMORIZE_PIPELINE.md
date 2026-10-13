# Memorize Pipeline Map

Maintained developer navigation for saved conversations, not a product manual or
repair log. MCP owns client-independent selection, admission, checkpoints and
recovery; memU owns extraction, memory storage and dossier operations. A category
is a dossier. Follow both sides of a boundary before changing either side.

## Reviewed Baseline

Static source trace, 2026-10-12:

| Repository | Reviewed HEAD (`buildfix`) |
| --- | --- |
| `mcp-memu-server` | `a88954469f8ff2ba58690ea5e2dd4491d15deca5` |
| `memu` | `3aeed83b8e5e0355513e7305ab5d0c381edcd4fd` |
| ST `memu-extension` (publication/polling seams) | `b4aae13bcf2a8d5c8d252e88ce7f1fe0bc3b3f0f` |
| ST `memu-plugin` (forwarding/status seams) | `c01a4adb8c4ed0eafec213840e36ed80bb97b8a3` |

The initial MCP trace used `ef2bf55`; the committed background-cursor change
through `a889544` was checked before publishing this map. Repair plans and audit
maps were navigation leads, not evidence of present behavior. No tests, services
or model calls were run for this documentation pass.

Later MCP changes through `51bbdf3` were traced at the changed seams below:
current background summaries, DM aliases, APImw publication and photo claims.
This updates those boundaries, not the original whole-map verification scope.
Provider-wait and media-failure seams below were additionally traced against
memU `1826793`, with fictional-provider and local-subprocess regression checks.
The WhatsApp policy-editor boundary was traced through OpenAlma `26685dd`;
Channels/MCP policy readers were unchanged and checked with disposable data.
Speaker attribution was traced through memU `050c3cc`, historical recall metadata
through `c83a969`, and reply identity/anchor enrichment through MCP `42375a0`.

## Route And Owners

```text
client-owned saved source
  -> MCP publication / supported source reader
  -> Soul-wide selection + captured filing policy + rolling context
  -> admitted conversation segment(s)
  -> memU routing + type extraction + prepared embeddings
  -> primary save WITH source checkpoint and pending/post-save markers
  -> dedupe -> dynamic dossier review
  -> pending Resource evidence -> consolidation when due
  -> failures retained by stage -> explicit Retry of unfinished work
```

The ordinary automatic/manual cross-conversation path and its save/recovery
callbacks are traced below. Historical imports reuse those seams with separate
state. This is not a claim that every public `memorize` variant, non-conversation
modality or client UI has identical admission and transaction behavior.

### Saved-Source Boundary

| Source owner | MCP read/publication seam | What must survive the boundary |
| --- | --- | --- |
| SillyTavern owns saved chat JSONL | `main.memorize` -> `conversation_sources.persist_sillytavern_history_snapshot` / `load_sillytavern_tail` | Scoped chat identity, original positions, full saved `source_message_count`, message role/name/time |
| WhatsApp owns originals; Hermes Channels owns the local source projection and routing/Mem settings | `conversation_sources.load_whatsapp_web_source_tail`, `load_whatsapp_channel_settings`; assistant attribution uses saved Hermes source speakers | Stable message key plus web-source row position, owner/peer/Soul evidence, chat identity, active-since floor; routing is not Mem policy |
| Atomic supplies its completed session transcript | `main.atomic_session_end` -> `persist_atomic_history_snapshot` / `load_atomic_tail` | Scoped session identity, transcript rows and activity recap; not a second memory engine |
| Iris supplies transcript events through Mentra ingress | `mentra_routes.mentra_transcripts_append` -> `persist_mentra_history_snapshot` / `load_mentra_tail` | Shared `mentra:iris` conversation, sitting/sequence evidence and saved event content; photo filing is a separate path |
| OpenAlma/Echo owns imported chat storage | `conversation_sources.import_source_path`, `load_import_tail`; `import_routes.register_import_routes` | Stored positions, immutable historical/current classification, source day, scoped chat and fixed registered history bound |
| MCP owns activity recaps | `activity_messages.record_activity_message` / `activity_message_rows`; callers include `free_turn._run_free_turn_chain` | One recap per completed free turn, not the underlying command/tool stream; activity row index is the checkpoint |

For ST, extension `src/memory/memorize.ts` owns `bindSavedSource`,
`withSavedSourceQueue` and `publishSavedSource`: publication reads `/api/chats/get`,
not the currently displayed unsaved chat. `buildTurnHistory` filters system/empty
rows but preserves original indices. `src/memory/exports.ts` saves replies before
`automatic` publication; deletion saves before `publish_only`. Plugin
`src/memu-endpoint.ts:proxyMemorizeConversation` forwards the source to MCP and
creates a local task ID only after accepted admission.

In `main.memorize`, ST snapshot publication precedes processing admission:
`publish_only` stops there; `automatic` may defer/coalesce; `manual` enters
`_memorize_owned`. Published source is not proof that extraction ran. Snapshot
writes use `conversation_sources._write_chat_snapshot` (temporary file + replace).
They are separate from memory-engine SQLite commits.

Iris transcript publication saves the snapshot, then a scoped identity receipt
under `transcripts/.published/`, before conversation state and acknowledgement.
The receipt survives loss of the individual chat folder. Both transcript readers
use `_read_mentra_snapshot`: missing history with a receipt is an unavailable
source, not a new empty chat. Strict Memorize refuses it; status reports degraded;
append and Start cannot silently recreate it or reset its sequence. Without
publication evidence, first-use chats still read empty; working thoughts alone
are not evidence. Reads never create receipts or upgrade old snapshots.
Snapshot and receipt saves are not one transaction: a failed receipt save leaves
the snapshot unacknowledged, and duplicate replay can finish publication. The
first-save crash interval, whole transcript-root loss and pre-repair history
without receipts remain outside this detection; duplicate replay still does not
rerun automatic Memorize eligibility.

Do not confuse **source ownership** with **speaker identity**. The shared user
and Soul scope do not make every imported `user` row the owner. See
`payload._label_owner_messages`, `conversation_sources._import_source_message`
and memU `memorize_speakers._build_speaker_map`: explicit `source_owner_id`, source
roles/references and established entity links carry attribution. Generic imported
roles alone do not assert owner identity. Preserve that evidence, not just prose.
The shared speaker builder does not resolve a participant from a name or alias.
The shared grouped renderer also never derives a group sender from `[Name]`
typed in the message body. Structured sender labels remain authoritative and
bracketed text stays literal, for replies, extraction and reflection alike.
Unlinked named participants retain their display labels in source context but
remain unidentified for deterministic memory attribution and dedupe. Explicit
source links and model-selected declared-relationship roster references remain
separate supported evidence; this repair does not rewrite old stored memories.
Extraction uses raw primary message bodies, not generated speaker headings, to
find mentioned declared people. In ambiguous contexts, copied display rows show
existing linked IDs or unlinked sender references; saved rows/labels stay intact.
Distinct cited source identities sharing a display label block implicit
assignment, whether linked or unresolved. All cited labels count, including
earlier names; repeated rows or renames for one identity do not collide.
The memory still saves without that link;
valid explicit Soul choices remain trusted, including absent-person subjects.

Reply entry validates identity selectors with the same resolver as retrieval,
before reading a Soul or calling a provider. Anchor enrichment normalizes the
identity-context role like the engine without changing caller-authored queries.
Fixed Atomic/Iris context timing and intentional exclusions are unchanged.
Iris working-state writes use the existing reply lock and recheck the sitting and
pause before saving. Intention preparation and lock admission share one 25-second
asynchronous deadline; a preparation/admission timeout is definitely unsaved, not
an ambiguous write result. Synchronous persistence still runs to completion.
Thoughts retain the shared 600-character cap and report truncation explicitly.
No write is automatically repeated by Iris after a lost result.
`main._annulment_write_session` commits intention removal and its prepared
reflection memories together for Iris and scheduled turns. Ordinary reply
publication uses the same transaction for Undo, working cache and consumed
context. Fresh Soul state is read inside that transaction; failures roll back
memory rows, search entries, reference allocation and state without cleanup
deletes. Removing an intention without a prepared reflection fails before writes.

## Selection And Context

Start in committed `app/main.py`: `_prepare_auto_memorize`,
`_build_cross_conversation_payload`, `_schedule_auto_memorize`,
`_memorize_owned`, `_prepare_frozen_memorize`. Automatic eligibility checks the
trigger's unmemorized sleep gap and primary token budget, then captures a
Soul-wide payload. Manual saved-ST processing also assembles cross-conversation
tails. `_prepare_frozen_memorize` subtracts fresh completed checkpoints without
refreshing captured rows; automatic coalescing retains a captured waiting payload.
ST cursor ceilings account for a saved chat shrinking during accepted work.
Before automatic free-slot admission, `_schedule_auto_memorize` removes explicitly
retired ST sources from captured rows, checkpoints, rolling summaries and ceilings.
Surviving sources retain their captured data and bounds. A deleted trigger keeps
its bookkeeping ID and already-established timing eligibility, but the surviving
primary token threshold still applies. Accepted/manual/Retry work is not filtered
again; missing original files alone do not invalidate valid frozen captures.

`app/services/cross_history.py` owns `_source_memorize_policy`,
`_load_cross_memorize_tails_from_sources`,
`_read_background_rolling_summaries_from_conversations` and source cursor
resolution. `_resolve_background_cursor` uses the later of the rolling-summary
checkpoint and effective completed Memorize checkpoint, including timestamp
floors. Turning Mem off must not replay already-memorized source rows. The same
boundary feeds background rollup, current context and saved-source Retry.
`conversation_sources.py` owns actual source projection/read bounds.
Ordinary processing excludes registered historical evidence; Echo owns that range.

OpenAlma `launcher/policy.py` owns policy-editor lookup and publication. Display
combines known DM aliases using the same routing/explicit-Mem precedence as the
Channels/MCP readers; missing Mem is defaulted only after aggregation. Saving
updates each existing alias row's policy/Mem while retaining its metadata. The
`app.policy_save` route stages the default and rows in one JSON write after
rejecting conflicting alias choices. Group-key normalization is preserved;
groups do not walk person mappings. Captured Retry filing policy is unchanged.

- **Primary / Mem-on:** `memorize_chat=True`; message-by-message evidence after
  the digest checkpoint, used for extraction provenance and allowed source days.
- **Mem-off:** `memorize_chat=False`; existing rolling prose plus a bounded new
  tail supplies supporting context, not independent primary extraction evidence.
  WhatsApp `full`/`listen_only` can each be Mem-on or Mem-off; `excluded` sources
  are skipped on fresh selection. A captured `filing_policy` overrides fresh policy
  during failure recovery.
- **Activities:** saved recaps are already summaries, but
  `activity_message_rows` marks them Mem-on. They participate in digest checkpoints
  and extraction without reconstructing tool streams or immediate free-turn
  Memorize. Mentra transcript selection is also always Mem-on at this baseline.

memU `src/memu/app/memorize_segments.py:_render_episode_with_background_context`
groups Mem-off tails **per chat**. Below `background_extra_messages_tokens`
(default 100 estimated tokens), keep a short real tail; at or above it, batch the
qualifying chats in one call with a separate summary for each. Preserve prior
rolling prose and readable source identity. This rule does not summarize primary
chats. `src/memu/utils/conversation.py` is the shared grouped display renderer.
Chat headings and import-app labels use the same single-line display normalization
as speaker labels, including current-chat overrides and free-turn origin headings.
Source identities, stored labels, message bodies and summary prose are unchanged.

Keep inline supporting context distinct from durable rolling state:
`main._run_background_rollup_for_conversation` separately prepares and conditionally
commits `rolling_summary` with its own cursor. Memorize primary publication does
not advance Mem-off rolling cursors. Its generated follow-up summaries are captured
with that segment, not installed as the next rolling summary.

## Extraction And Commit Boundaries

MCP `app/services/memorize_endpoint.py:memorize_endpoint` allocates segment ranges
and schedules `run_memorize_segments`. The worker prepares saved segment files and
calls memU `src/memu/app/memorize.py:memorize_segments_batch`, supplying
`on_segment_saved` and `on_dedupe_complete`. A cross-conversation batch uses one
segment job; other conversation calls can produce several sequential units.

`memorize_segments_batch` separates primary/background rows, renders supporting
context once, routes episodes (`_route_segment`), selects dossier context and
extracts applicable memory types. Dates are validated against selected **primary**
source days, not broadened by background-only dates. The default type registry is
`src/memu/prompts/memory_type/__init__.py`; speaker attribution is in
`memorize_speakers.py`. Preparation/model work is not the primary commit.
`memorize_parsing._parse_memory_type_response_xml` preserves literal less-than
signs inside content, replacement facts, entity names/types and category labels,
including `a<b`, without repairing structural XML. Extraction retains unknown
entity references as literal text; dossier/reflection parsers reject them.
Quote-aware tag shielding leaves attributes to XML validation. CDATA, comments
and processing instructions do not change content nesting; broken structure still
fails before persistence. Weekly companion prose uses all child/tail text when
preparing its embedding and final write, not just the first XML text node.

The important naming trap: **`_memorize_categorize_items` performs the primary
save; `_memorize_persist_and_index` performs post-save dynamic dossier review.**
The traced conversation batch invokes these per extracted source unit, not in
one all-or-nothing transaction for the whole request.

| Boundary | Owner and durable result |
| --- | --- |
| Before primary save | `_prepare_plan` prepares embeddings/supersession work. MCP segment files/manifest exist, but no committed Resource yet proves acceptance. Preparation may repeat on Retry. |
| Primary SQLite save | `_memorize_categorize_items` -> `_process_plan` writes Resource, episode/generalized items, filing links/candidates and associated triples. Before `session.commit`, `on_segment_saved` calls MCP `publish_job` in that **same session**: primary source cursors, pending segment ID and `segment_work[segment_id]="dedupe"` become durable with those memories. Historical cursor/pending work goes into `import_state` instead. |
| Post-save dedupe | `memorize_dedupe._memorize_dedupe_merge` commits merges and `on_dedupe_complete`'s MCP `"review"` phase together in its own transaction. The primary save already exists. |
| Post-save dynamic dossier review | `_memorize_persist_and_index` prepares/generates/applies candidate decisions, committing each applied decision. MCP clears completed phase work only after successful postprocessing. A later failure does not undo earlier saves or review decisions. |

Correction preparation fails on an invalid old-fact embedding before the primary
save; a valid vector with no matching fact still creates the new memory normally.
Memory deletion removes graph edges by their current memory endpoints, not by
historical `source_memory_id`. Deleting a merged-away predecessor cannot erase a
mention now owned by its survivor; deleting that survivor still removes the edge.

For these normal conversation calls, `_process_plan` upgrades the segment file to
an envelope containing `content`, the exact rendered `segment_background_context`
and `segment_background_conversation_ids`. The complete accepted message list
retains its row count; summaries are **not fake indexed messages**. Resource
`local_path` owns that file; Resource also captures selected prior-context IDs.
Changing the envelope, range allocation or callback contract requires checking
the later Resource reader, not only extraction prompts.

Files and SQLite are not a single transaction. `run_memorize_segments`'s
`release_uncommitted` removes files/manifest reservations without committed
Resources on handled failure, while keeping committed evidence. That cleanup is
not a hard-kill recovery guarantee. Do not generalize the conversation callback
boundary to image filing: memU's public image path has early Resource creation.

## Consolidation

MCP `app/services/consolidation.py:gather_consolidation_inputs` gathers pending
work across the Soul, refusing unfinished Memorize postprocessing. It resolves
each selected segment through scoped **Resource ownership**, reads its saved
file and checks the segment's exact row count. Missing/ambiguous/unreadable
ownership fails rather than silently dropping evidence. It uses active selected
memories and Resource-owned prior context, not mutable current retrieval history.

`_run_consolidation_pipeline_once` retains captured supporting blocks and may add
fresh cross-conversation context. Fresh Mem-off tails use the same bounded
renderer after unpaid time/size admission; already captured raw rows are excluded
from that supplemental work. Saved supporting blocks and current rolling prose
coexist, so later rollup progress is not lost. Import sizing and prefix shrinking
include current summary rows without loading live raw tails.
`cross_history._format_all_chat_history_for_ai(memory_context=True)` keeps
raw Mem-off cross-tail rows out of the primary history. Thus Resource evidence
is durable, but the entire later prompt is not claimed to be a frozen snapshot.

The model stages are dossier revision, identity/anchor maintenance and ordinary
weekly reflection. Their publication boundaries are separate from those model
stages:

1. `prepare_dossier_consolidation_context` batches ordinary dossier revisions,
   then applies them via memU `dossier.apply_dossier_revision`. Each successful
   dossier revision is durable independently of the later stages.
2. `run_consolidation_llm` prepares identity/anchors/narrative/life-goal changes;
   ordinary processing also generates weekly intentions, edges and companion
   reflection. This generation itself does not publish the final state.
3. `write_consolidation_outputs` commits anchors, narrative/history, life goals,
   ordinary reflection outputs and subtraction of **only selected pending IDs**
   together. It merges intentions against fresh state and preserves newly queued
   work. Dossier/narrative journal finishing follows the database commit.

Shared `xml_utils.extract_xml_fragment` shields opaque XML/root boundaries and
uses memU `utils.xml.escape_xml_prose` for bare prose punctuation. The same
normalizer covers single/batch dossier revision and dynamic dossier creation.
Only each parser's named prose fields are normalized, not attributes or IDs.
Tag candidates require a complete name boundary, so comparisons containing
apostrophes are not mistaken for quoted attributes. Tags and entity references remain strict;
scalar edge fields reject children rather than discarding nested text. Complete
tag-shaped literal prose still requires entities or CDATA; this is not XML salvage.

Ordinary consolidation is interval/size-gated; a `not_due` result leaves pending
evidence durable. Historical Echo consolidation uses bounded whole segments,
maintains dossiers/identity, skips weekly reflection and does not advance the
ordinary consolidation clock. Its cursor/stage/error/pending IDs stay in
`import_state`, separate from ordinary digest progress.

Dossier scheduling treats prose citations as dependencies even without an
attached membership. Ordinary updates include inactive dependencies outside the
current chat segments; historical selection and pending-import exclusions still
bound review. Revision bundles supply their active successors independently of
search `top_k`, so the Soul can repair citations in submitted prose. Exact memory
lookups do not redirect. An excluded old citation can remain until its import is
processed; retaining it uses the existing citation-to-membership rule without
marking that evidence reviewed. Empty dossiers keep the Soul's submitted prose,
not an automatically substituted copy of their previous text.

## Failure And Retry Ownership

### Atomic Edit Admission

Stored-memory merging occurs only in Memorize dedupe, including image filing
and dedupe-phase Retry. Supported MCP entrypoints retain the Soul-wide claim
through correction preparation, saving and dedupe; conversation IDs do not
create separate writers. Reply/intention and APImw publication can overlap but
only create memories, not merge correction targets. These process-local claims
do not serialize direct engine/repository use or a second server process.

The shared HTTP gate refuses Atomic memU mutations while that Soul has a
Memorize/import/photo claim or a running consolidation. Memory and dossier
reviews, approvals, deletion, memberships, entities and relationships share this
admission; GET reads and independent Atomic Workspace notes do not.

An admitted edit holds the existing claim until its handler finishes, preventing
a memory job from starting during paid edit preparation. `None` distinguishes an
edit or photo from ordinary (`False`) or historical (`True`) memory processing, so they
cannot unpause interrupted recovery. Existing claim release rechecks queued
automatic work even if the edit fails. Snapshot checks and dead tabs remain:
unsaved drafts can still become stale while memory processing runs.

### Provider Waits

memU's existing HTTP clients bound each call, including throttle/retry waits,
with the operation's total elapsed timeout. Chat remains single-attempt; vision
and HTTP-client embeddings retry only connection failures or an explicit 429
`Retry-After`, never an ambiguous read/protocol failure or 5xx. Gemini text/media
embeddings and transcription have total deadlines too. Completion parsing rejects
empty or unsupported responses before success hooks run. Audio/video preprocessing
lets provider failures escape; local file/ffmpeg fallbacks remain unchanged.
Failures propagate to callers; deadlines do not roll back committed evidence.

`ClaudeCLIClient` directly awaits its subprocess and kills/reaps the direct child
on task cancellation or its configured timeout, retaining prompt-file cleanup.
MCP's flag-based Cancel admission/boundary is unchanged; it does not become an
immediate interruption of accepted work. Child cleanup cannot undo remote work
already accepted and does not promise descendant-process termination.

### Reply-Adjacent Work

APImw `_turn_launch_apimw` captures the database generation before task launch.
`_apimw_persist` publishes valid context and `apimw_subconscious_insight` under a
short Soul lock, then embeds outside it. Archival item creation and error writes
recheck that generation, retirement and pause state; consumed insights are never
restored by archival completion. State and item commits remain independent, so
embedding failure/cancellation can leave published state without an archival item.

`conversations.prior_context` holds the ordered pending selection as JSON
references, not finished memory prose. APImw retains selected IDs, optional graph
presentation references and procedural `(domain, id)` references. Shared
`apimw.render_prior_context` fetches only current active rows for that owner/Soul,
independently of the new query's filters; procedural references use a read-only
exact sidecar lookup. Missing/deleted/merged/superseded rows are omitted, not
replaced or re-searched. Current memory/seed prose and relative dates are rendered
at prompt construction for retrieval output and import allowance estimation.
Already prepared `/turn` prompts remain caller-authored snapshots.

Retrieval and failed/dry-run replies do not consume this selection; successful
turn publication clears it. Undo still does not restore it. The separate
Soul-wide prior-context ID history and Resource-owned consolidation evidence are
unchanged. The state PATCH boundary accepts only normalized reference lists or
null. There is no runtime legacy-prose reader or second pending-context field.

Before starting this code against existing databases, back up and run the
explicit release upgrade. `2026-10-12-prior-context-references` expires legacy
queued prose once, as Marcos approved, while preserving valid current references
and all memories, chat/Undo state and reflection evidence. The migration ledger
prevents repeat expiration. Source repair tests did not apply it to live DBs.

Iris photo workers retain `_image_finalize_tasks` FIFO/coalescing and reserve
the existing memory-work claim as `None` before native image Memorize, so a photo
cannot hide interrupted Memorize or its Retry. Provider waits hold
no reply lock; `finally` releases the claim and preserves queued automatic work
even after photo failure. Completed-Memorize Retry reserves that same claim before
dispatching follow-on reflection. Claim release keeps waiting captures while
durable Memorize/consolidation recovery remains required; it does not bypass Retry.
`_finish_memorize_claim` clears stale active
progress and Cancel before handing off a fresh job. These are process-local
claims, not restart recovery; unfinished photo filing is not automatically resumed.
Photo finalization checks registered work before saved memory rows. An explicit
re-finalize joins that worker and selects the latest validated sitting to receive
its outcome; merely starting a replacement sitting does not subscribe it. Error
delivery requires that observer's active lease, and success clears only that
photo's warning. Queue acceptance and saved-item duplicate evidence do not assert
that later dedupe/review completed; no full background-filing Retry is added.

Live `_MEMORIZE_PROGRESS`, busy claims and cancellation flags are process-local
status, **not a durable job queue**. ST reads progress by owner/Soul without a
plugin routing cache and discovers active work even without local metadata.
Polling is serialized; scope/generation/token checks discard stale responses.
Unknown status preserves an active observation, not a failed-work flag; known
idle/skipped/cancelled outcomes clear it without completion sync. Cancel success
acknowledges a request, so the banner waits for terminal progress. A local task
ID or SUCCESS is not the engine's evidence/checkpoint store or a per-chat receipt.

| Unfinished work | Durable owner and recovery entry |
| --- | --- |
| Ordinary extraction/save/post-save work | Soul `memorize_failure` holds targets, captured filing policy and segment phases. `main.retry_memorize` (`/memorize/retry`) -> `import_routes.run_waiting_memorize` / `recover_memorize`. memU `resume_memorize_segment` resumes committed dedupe/review **without extraction**. Remaining unsaved work is rebuilt from saved sources by `_saved_memorize_payload` under captured filing policy; pre-save model work may repeat. |
| Ordinary consolidation | Pending Resources/IDs plus Soul `last_consolidation_error` / error time. `main.retry_consolidation` runs forced consolidation; committed dossier revisions remain. Pending IDs are consumed by final publication, not cleared on failure. |
| Historical import | `import_state` owns historical cursor, pending IDs, segment phases, stage/error and `ordinary_waiting`. `import_routes.run_import_batch` / `run_import` and `/imports/retry` recover historical work; completed history hands waiting ordinary work to `run_waiting_memorize`. Ordinary Retry refuses unfinished import history. |
| Independent background rollup | Conversation rolling summary/cursor plus `last_background_error`; `_run_background_rollup_for_conversation` owns conditional writes. Do not treat this as a committed primary segment or invent a separate segment-recovery ledger for it. |

`state.write_conversation_state` supports caller-owned connections;
`state.patch_memorize_failure` preserves fresh phases/targets and refuses clearing
unfinished postprocessing. `soul_state.activity_pause` plus
`main._soul_activity_pause` expose unresolved Memorize/consolidation/import failure
to activity admission. A dormant interrupted-run marker requires explicit Retry;
there is no automatic durable task resumption after restart.

ST source retirement is another boundary, not memory deletion:
`main.retire_conversation_source` / `conversation_sources.retire_sillytavern_source`
exclude fresh source reads and clear rolling state while leaving accepted work,
filed memories and Resource recovery intact. Follow checkpoint completion and
Retry anchoring before changing retirement behavior.

## Upkeep And Verification Scope

When changing a stage, review its immediate producer, consumer and failure
callback; update this map and the reviewed pins when that boundary changes.
Keep function names and ownership here, not line numbers, endpoint inventories,
repair status or prompt copies. Re-trace changed working-tree source against an
explicit revision before adopting it as maintained fact.

The described normal-call ownership and commit ordering are source-traced; no
unresolved policy choice is inferred from old audit recommendations. Runtime
failure/Retry, restart behavior, provider responses and filesystem interruption
were not exercised. Other client UIs and non-conversation modalities need their
own trace before extending these claims. Update the relevant producer/consumer
seams and provenance together when later commits change this flow.
