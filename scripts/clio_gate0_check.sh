#!/usr/bin/env bash
set -euo pipefail

SCRIPT_NAME="$(basename "$0")"

usage() {
  cat <<'USAGE'
CLIO Gate-0 live fold-step-0 check

Runs before any CLIO Phase-2 candidate fold/retire operation. This script
does not merge, cherry-pick, push, restart, mutate services, or edit runtime
state. It only validates the fold packet's V4 provenance.

Required environment:
  OPENCLAW_REPO
  REPORTS_DIR
  PROMPTS_DIR
  CANDIDATE_NAME
  CANDIDATE_REF
  CANDIDATE_SHA
  TARGET_REF
  TARGET_HEAD
  V4_ARTIFACT
  V4_SHA256
  V4_AUTHORITY_KIND              manifest|pasteback
  V4_AUTHORITY_ARTIFACT
  V4_AUTHORITY_SHA256
  GATE0_VERDICT_LINE
  GATE0_AUTHORITY_LINE

For GATE0-VERDICT: GO-WITH-FIX, also required:
  GATE0_FIX_EVIDENCE_ARTIFACT
  GATE0_FIXES_SATISFIED_LINE

Exit codes:
  0   PASS: Gate-0 authorized
  42  BLOCK: Gate-0 failed closed
USAGE
}

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
  usage
  exit 0
fi

sha256_of() {
  sha256sum "$1" | awk '{print $1}'
}

fail_block() {
  echo "BLOCK: $*" >&2
  exit 42
}

require_input() {
  local name="$1"
  if [ -z "${!name:-}" ]; then
    fail_block "missing required input: $name"
  fi
}

require_contains() {
  local haystack="$1"
  local needle="$2"
  local label="$3"
  case "$haystack" in
    *"$needle"*) ;;
    *) fail_block "$label missing expected token: $needle" ;;
  esac
}

git_resolve() {
  git -c "safe.directory=$OPENCLAW_REPO" -C "$OPENCLAW_REPO" rev-parse "$1"
}

manifest_lines_for_candidate() {
  grep -F 'GATE0-MANIFEST:' "$V4_AUTHORITY_ARTIFACT" \
    | grep -F "candidate=$CANDIDATE_NAME" \
    | grep -F "ref=$CANDIDATE_REF" \
    | grep -F "sha=$CANDIDATE_SHA" || true
}

authority_represents_artifact() {
  local artifact_path="$1"
  local artifact_sha="$2"
  [ "$V4_AUTHORITY_KIND" = "manifest" ] \
    && grep -F "artifact=$artifact_path" "$V4_AUTHORITY_ARTIFACT" \
      | grep -Fq "artifact_sha256=$artifact_sha"
}

main() {
  local required=(
    OPENCLAW_REPO
    REPORTS_DIR
    PROMPTS_DIR
    CANDIDATE_NAME
    CANDIDATE_REF
    CANDIDATE_SHA
    TARGET_REF
    TARGET_HEAD
    V4_ARTIFACT
    V4_SHA256
    V4_AUTHORITY_KIND
    V4_AUTHORITY_ARTIFACT
    V4_AUTHORITY_SHA256
    GATE0_VERDICT_LINE
    GATE0_AUTHORITY_LINE
  )
  local name
  for name in "${required[@]}"; do
    require_input "$name"
  done

  [ -d "$OPENCLAW_REPO" ] || fail_block "OPENCLAW_REPO missing directory: $OPENCLAW_REPO"
  [ -d "$REPORTS_DIR" ] || fail_block "REPORTS_DIR missing directory: $REPORTS_DIR"
  [ -d "$PROMPTS_DIR" ] || fail_block "PROMPTS_DIR missing directory: $PROMPTS_DIR"

  [ -f "$V4_ARTIFACT" ] || fail_block "missing V4 artifact: $V4_ARTIFACT"
  local actual_sha256
  actual_sha256="$(sha256_of "$V4_ARTIFACT")"
  [ "$actual_sha256" = "$V4_SHA256" ] \
    || fail_block "V4 artifact hash mismatch expected=$V4_SHA256 actual=$actual_sha256"

  [ -f "$V4_AUTHORITY_ARTIFACT" ] || fail_block "missing V4 authority artifact: $V4_AUTHORITY_ARTIFACT"
  local actual_authority_sha256
  actual_authority_sha256="$(sha256_of "$V4_AUTHORITY_ARTIFACT")"
  [ "$actual_authority_sha256" = "$V4_AUTHORITY_SHA256" ] \
    || fail_block "V4 authority hash mismatch expected=$V4_AUTHORITY_SHA256 actual=$actual_authority_sha256"

  case "$V4_AUTHORITY_KIND" in
    manifest|pasteback) ;;
    *) fail_block "invalid V4 authority kind: $V4_AUTHORITY_KIND" ;;
  esac

  if [ "$V4_AUTHORITY_KIND" = "manifest" ]; then
    case "$GATE0_AUTHORITY_LINE" in
      "GATE0-MANIFEST: "*) ;;
      *) fail_block "manifest authority kind requires a GATE0-MANIFEST authority line" ;;
    esac
  else
    case "$GATE0_AUTHORITY_LINE" in
      "GATE0-PASTEBACK: "*) ;;
      "GATE0-MANIFEST: "*) fail_block "pasteback authority cannot masquerade as manifest authority" ;;
      *) fail_block "pasteback authority kind requires a GATE0-PASTEBACK authority line" ;;
    esac
  fi

  git_resolve "$CANDIDATE_REF" | grep -Fxq "$CANDIDATE_SHA" \
    || fail_block "candidate ref does not resolve to reviewed sha"
  git_resolve "$TARGET_REF" | grep -Fxq "$TARGET_HEAD" \
    || fail_block "target ref does not match packet target head"

  grep -Fxq "$GATE0_VERDICT_LINE" "$V4_ARTIFACT" \
    || fail_block "exact verdict line absent from V4 artifact"
  local verdict_value
  case "$GATE0_VERDICT_LINE" in
    "GATE0-VERDICT: GO | "*) verdict_value="GO" ;;
    "GATE0-VERDICT: GO-WITH-FIX | "*) verdict_value="GO-WITH-FIX" ;;
    *) fail_block "verdict is not affirmative GO/GO-WITH-FIX" ;;
  esac
  require_contains "$GATE0_VERDICT_LINE" "candidate=$CANDIDATE_NAME" "verdict line"
  require_contains "$GATE0_VERDICT_LINE" "ref=$CANDIDATE_REF" "verdict line"
  require_contains "$GATE0_VERDICT_LINE" "sha=$CANDIDATE_SHA" "verdict line"

  local verdict_lines
  verdict_lines="$(grep -F 'GATE0-VERDICT:' "$V4_ARTIFACT" \
    | grep -F "candidate=$CANDIDATE_NAME" \
    | grep -F "ref=$CANDIDATE_REF" \
    | grep -F "sha=$CANDIDATE_SHA" || true)"
  local verdict_count
  verdict_count="$(printf '%s\n' "$verdict_lines" | sed '/^$/d' | wc -l | tr -d ' ')"
  [ "$verdict_count" = "1" ] \
    || fail_block "expected exactly one canonical verdict line, found $verdict_count"
  if printf '%s\n' "$verdict_lines" | grep -Eiq '^GATE0-VERDICT: (HOLD|REVERT|ABORT|NO-FOLD|NO-GO|BLOCK)( |$|\|)'; then
    fail_block "competing blocking verdict line exists in cited V4 artifact"
  fi

  grep -Fxq "$GATE0_AUTHORITY_LINE" "$V4_AUTHORITY_ARTIFACT" \
    || fail_block "exact V4 authority line absent"
  require_contains "$GATE0_AUTHORITY_LINE" "candidate=$CANDIDATE_NAME" "authority line"
  require_contains "$GATE0_AUTHORITY_LINE" "ref=$CANDIDATE_REF" "authority line"
  require_contains "$GATE0_AUTHORITY_LINE" "sha=$CANDIDATE_SHA" "authority line"
  require_contains "$GATE0_AUTHORITY_LINE" "artifact=$V4_ARTIFACT" "authority line"
  require_contains "$GATE0_AUTHORITY_LINE" "artifact_sha256=$V4_SHA256" "authority line"
  require_contains "$GATE0_AUTHORITY_LINE" "verdict=$verdict_value" "authority line"

  if [ "$V4_AUTHORITY_KIND" = "manifest" ]; then
    local manifest_matches
    manifest_matches="$(manifest_lines_for_candidate)"
    local blocking_manifest_lines
    blocking_manifest_lines="$(printf '%s\n' "$manifest_matches" | grep -E '(^|[|] )verdict=(HOLD|REVERT|ABORT|NO-FOLD|NO-GO|BLOCK)([ |]|$)' || true)"
    if [ -n "$blocking_manifest_lines" ]; then
      local superseding_go_lines
      superseding_go_lines="$(printf '%s\n' "$manifest_matches" | grep -F 'verdict=GO' | grep -F 'supersedes_artifact=' || true)"
      [ -n "$superseding_go_lines" ] \
        || fail_block "V4 authority manifest contains blocking ruling without explicit GO supersession"
    fi
  else
    if grep -Fq 'supersedes_artifact=' "$V4_AUTHORITY_ARTIFACT"; then
      fail_block "paste-back authority cannot prove automated supersession"
    fi
    if manifest_lines_for_candidate | grep -Eiq '(^|[|] )verdict=(HOLD|REVERT|ABORT|NO-FOLD|NO-GO|BLOCK)([ |]|$)'; then
      fail_block "paste-back authority contains blocking ruling but cannot prove supersession"
    fi
  fi

  local path path_sha
  while IFS= read -r -d '' path; do
    if grep -Fq "$CANDIDATE_NAME" "$path" || grep -Fq "$CANDIDATE_SHA" "$path"; then
      if grep -Eiq 'GATE0-VERDICT: (HOLD|REVERT)|HOLD|REVERT|ABORT|SUPERSEDE|NO-FOLD|NO-GO|BLOCK' "$path"; then
        path_sha="$(sha256_of "$path")"
        if ! authority_represents_artifact "$path" "$path_sha"; then
          fail_block "possible blocking V4 artifact not represented in authority manifest: $path"
        fi
      fi
    fi
  done < <(find "$REPORTS_DIR" "$PROMPTS_DIR" -maxdepth 1 -type f \( -iname '*V4*CLIO*.md' -o -iname '*PP_V4_CLIO*.md' \) -print0)

  if printf '%s\n' "$GATE0_VERDICT_LINE" | grep -Fq 'GATE0-VERDICT: GO-WITH-FIX'; then
    require_input GATE0_FIX_EVIDENCE_ARTIFACT
    require_input GATE0_FIXES_SATISFIED_LINE
    [ -f "$GATE0_FIX_EVIDENCE_ARTIFACT" ] || fail_block "GO-WITH-FIX fix evidence artifact absent"
    grep -Fxq "$GATE0_FIXES_SATISFIED_LINE" "$GATE0_FIX_EVIDENCE_ARTIFACT" \
      || fail_block "exact fixes-satisfied line absent"
    local verdict_fixes evidence_fixes
    verdict_fixes="$(printf '%s\n' "$GATE0_VERDICT_LINE" | sed -n 's/.*| fixes=\([^|]*\).*/\1/p')"
    evidence_fixes="$(printf '%s\n' "$GATE0_FIXES_SATISFIED_LINE" | sed -n 's/.*| fixes=\([^|]*\).*/\1/p')"
    [ -n "$verdict_fixes" ] || fail_block "GO-WITH-FIX verdict has empty fixes list"
    [ "$verdict_fixes" = "$evidence_fixes" ] \
      || fail_block "fixes-satisfied list differs from verdict fixes list"
    require_contains "$GATE0_FIXES_SATISFIED_LINE" "candidate=$CANDIDATE_NAME" "fixes-satisfied line"
    require_contains "$GATE0_FIXES_SATISFIED_LINE" "ref=$CANDIDATE_REF" "fixes-satisfied line"
    require_contains "$GATE0_FIXES_SATISFIED_LINE" "sha=$CANDIDATE_SHA" "fixes-satisfied line"
  fi

  echo "PASS: Gate-0 authorized"
}

main "$@"
