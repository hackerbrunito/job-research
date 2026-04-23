"""Triage page — review unique job titles and mark them accept/reject/unsure."""

from __future__ import annotations

from typing import Final

import pandas as pd
import streamlit as st

from job_research.app.common import (
    Profile,
    _norm_title,
    apply_title_labels_to_judged,
    connect,
    delete_title_label,
    get_triage_candidates,
    list_profiles,
    list_title_labels,
    read_only_connection,
    save_title_label,
)
from job_research.logging_setup import get_logger

log = get_logger(__name__)

# ---- Page-local constants --------------------------------------------------
_SESSION_KEY_ACTIVE_PROFILE: Final[str] = "active_profile_id"
_LABEL_OPTIONS: Final[list[str]] = ["", "accept", "reject", "unsure"]
_TABLE_HEIGHT_PX: Final[int] = 500


# ---- Helpers ---------------------------------------------------------------


def _load_profiles_safely() -> list[Profile]:
    try:
        with read_only_connection() as con:
            return list_profiles(con)
    except Exception as exc:
        log.warning("triage.profiles.load_failed", error=str(exc))
        return []


def _load_candidates(profile_id: str, *, include_decided: bool) -> pd.DataFrame:
    try:
        with read_only_connection() as con:
            return get_triage_candidates(
                con, profile_id, include_decided=include_decided
            )
    except Exception as exc:
        log.warning("triage.candidates.load_failed", error=str(exc))
        return pd.DataFrame()


def _load_title_labels(profile_id: str) -> list[dict]:
    try:
        with read_only_connection() as con:
            return list_title_labels(con, profile_id)
    except Exception as exc:
        log.warning("triage.labels.load_failed", error=str(exc))
        return []


def _compute_metrics(profile_id: str) -> dict[str, int]:
    """Return summary counts for the metrics row."""
    try:
        with read_only_connection() as con:
            result = con.execute(
                """
                SELECT
                    COUNT(DISTINCT lower(trim(job_title)))                     AS total_staged,
                    COUNT(DISTINCT CASE WHEN ensemble_verdict = 'accept'
                                        THEN lower(trim(job_title)) END)       AS accepted,
                    COUNT(DISTINCT CASE WHEN ensemble_verdict = 'reject'
                                        THEN lower(trim(job_title)) END)       AS rejected
                FROM judged_job_offers
                WHERE profile_id = ?
                """,
                [profile_id],
            ).fetchone()
            # pending: ensemble_verdict='review' and no user label
            pending_row = con.execute(
                """
                SELECT COUNT(DISTINCT lower(trim(j.job_title)))
                FROM judged_job_offers j
                LEFT JOIN profile_title_labels l
                       ON l.profile_id = j.profile_id
                      AND l.title_norm  = lower(trim(j.job_title))
                WHERE j.profile_id = ?
                  AND j.ensemble_verdict = 'review'
                  AND l.label IS NULL
                """,
                [profile_id],
            ).fetchone()
    except Exception as exc:
        log.warning("triage.metrics.load_failed", error=str(exc))
        return {"total_staged": 0, "accepted": 0, "rejected": 0, "pending": 0}

    if result is None:
        return {"total_staged": 0, "accepted": 0, "rejected": 0, "pending": 0}

    return {
        "total_staged": int(result[0] or 0),
        "accepted": int(result[1] or 0),
        "rejected": int(result[2] or 0),
        "pending": int(pending_row[0] or 0) if pending_row else 0,
    }


def _build_editor_df(candidates: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of candidates with a 'label' column ready for data_editor."""
    if candidates.empty:
        return pd.DataFrame(
            columns=[
                "display_title",
                "company_sample",
                "count",
                "rule_verdict",
                "ensemble_verdict",
                "label",
            ]
        )
    df = candidates.copy()
    # Use existing user_label as the starting value for the editable column,
    # but present '' when there's no label yet.
    df["label"] = df["user_label"].fillna("").astype(str)
    return df[
        [
            "display_title",
            "company_sample",
            "count",
            "rule_verdict",
            "ensemble_verdict",
            "label",
        ]
    ]


# ---- Rendering helpers -----------------------------------------------------


def _render_metrics(profile_id: str) -> None:
    metrics = _compute_metrics(profile_id)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total staged", metrics["total_staged"])
    c2.metric("Accepted", metrics["accepted"])
    c3.metric("Rejected", metrics["rejected"])
    c4.metric("Pending review", metrics["pending"])


def _render_candidate_editor(
    editor_df: pd.DataFrame,
    *,
    show_all: bool,
) -> pd.DataFrame:
    """Render st.data_editor and return the (possibly modified) DataFrame."""
    if editor_df.empty:
        st.info("No candidates found. Run the pipeline first to populate staged jobs.")
        return editor_df

    column_config = {
        "display_title": st.column_config.TextColumn(
            "Title", disabled=True, width="large"
        ),
        "company_sample": st.column_config.TextColumn(
            "Company (sample)", disabled=True, width="medium"
        ),
        "count": st.column_config.NumberColumn("Seen", disabled=True, width="small"),
        "rule_verdict": st.column_config.TextColumn(
            "Rule verdict", disabled=True, width="small"
        ),
        "ensemble_verdict": st.column_config.TextColumn(
            "System verdict", disabled=True, width="small"
        ),
        "label": st.column_config.SelectboxColumn(
            "Your label",
            options=_LABEL_OPTIONS,
            required=False,
            width="small",
        ),
    }

    edited = st.data_editor(
        editor_df,
        column_config=column_config,
        use_container_width=True,
        hide_index=True,
        height=_TABLE_HEIGHT_PX,
        key="triage_editor",
        num_rows="fixed",
    )
    return edited


def _render_batch_actions(editor_df: pd.DataFrame) -> pd.DataFrame:
    """Render batch-action buttons. Returns a new df with labels possibly updated."""
    col_accept, col_reject, col_clear = st.columns(3)

    with col_accept:
        accept_all = st.button(
            "Accept all review",
            help="Set label to 'accept' for every row where rule_verdict is 'review'",
        )
    with col_reject:
        reject_all = st.button(
            "Reject all review",
            help="Set label to 'reject' for every row where rule_verdict is 'review'",
        )
    with col_clear:
        clear_labels = st.button(
            "Clear my labels",
            help="Reset the label column to blank for all visible rows",
        )

    if editor_df.empty:
        return editor_df

    # Read whatever is currently in session_state for the editor
    current = st.session_state.get("triage_editor", {})
    edited_rows: dict = (
        current.get("edited_rows", {}) if isinstance(current, dict) else {}
    )

    if accept_all or reject_all or clear_labels:
        df = editor_df.copy()
        for i, row in df.iterrows():
            idx = int(i)  # type: ignore[arg-type]
            if accept_all and row.get("rule_verdict") == "review":
                df.at[i, "label"] = "accept"
                edited_rows[idx] = {"label": "accept"}
            elif reject_all and row.get("rule_verdict") == "review":
                df.at[i, "label"] = "reject"
                edited_rows[idx] = {"label": "reject"}
            elif clear_labels:
                df.at[i, "label"] = ""
                edited_rows[idx] = {"label": ""}
        # Push the mutation back into session_state so data_editor picks it up
        if "triage_editor" in st.session_state:
            st.session_state["triage_editor"]["edited_rows"] = edited_rows
        return df

    return editor_df


def _render_save_apply(
    edited_df: pd.DataFrame,
    original_df: pd.DataFrame,
    profile_id: str,
) -> None:
    """Save labels and apply them to judged results."""
    if st.button("Save labels and apply to results", type="primary"):
        if edited_df.empty:
            st.warning("Nothing to save.")
            return

        saved_count = 0
        errors: list[str] = []

        try:
            with connect(read_only=False) as con:
                for _, row in edited_df.iterrows():
                    new_label = str(row.get("label") or "").strip()
                    if not new_label:
                        continue

                    # Compare against original to detect changes
                    title_n = _norm_title(str(row.get("display_title") or ""))
                    orig_match = (
                        original_df[
                            original_df["display_title"] == row["display_title"]
                        ]
                        if not original_df.empty
                        else pd.DataFrame()
                    )
                    orig_label = ""
                    if not orig_match.empty:
                        orig_label = str(
                            orig_match.iloc[0].get("user_label") or ""
                        ).strip()

                    if new_label != orig_label:
                        try:
                            save_title_label(
                                con,
                                profile_id=profile_id,
                                title_norm=title_n,
                                label=new_label,
                            )
                            saved_count += 1
                        except ValueError as exc:
                            errors.append(str(exc))

                n_updated = apply_title_labels_to_judged(con, profile_id)

        except Exception as exc:
            log.error("triage.save.failed", error=str(exc))
            st.error(f"Failed to save labels: {exc}")
            return

        # Refresh transform so fact table reflects new verdicts
        try:
            from job_research.transform import run_transform

            run_transform()
        except Exception as exc:
            log.warning("triage.transform.failed", error=str(exc))

        st.cache_data.clear()

        if errors:
            st.warning(f"Some labels skipped (invalid): {'; '.join(errors)}")

        st.success(
            f"Labels saved. {n_updated} title(s) updated. Results page refreshed."
        )
        log.info(
            "triage.labels.saved",
            profile_id=profile_id,
            saved=saved_count,
            applied=n_updated,
        )


def _render_label_dictionary(profile_id: str) -> None:
    """Collapsible expander showing all saved labels with individual delete."""
    with st.expander("Your saved labels for this profile", expanded=False):
        labels = _load_title_labels(profile_id)
        if not labels:
            st.caption("No labels saved yet.")
            return

        labels_df = pd.DataFrame(labels)
        st.dataframe(
            labels_df[["title_norm", "label", "count_seen", "updated_at"]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "title_norm": st.column_config.TextColumn("Title (normalised)"),
                "label": st.column_config.TextColumn("Label"),
                "count_seen": st.column_config.NumberColumn("Times seen"),
                "updated_at": st.column_config.DatetimeColumn("Last updated"),
            },
        )

        st.caption("Delete a saved label:")
        title_options = [row["title_norm"] for row in labels]
        to_delete = st.selectbox(
            "Select title to delete",
            options=["", *title_options],
            key="triage_delete_select",
        )
        if st.button("Delete label", key="triage_delete_btn") and to_delete:
            try:
                with connect(read_only=False) as con:
                    delete_title_label(con, profile_id=profile_id, title_norm=to_delete)
                st.cache_data.clear()
                st.success(f"Label for '{to_delete}' deleted.")
                log.info(
                    "triage.label.deleted",
                    profile_id=profile_id,
                    title_norm=to_delete,
                )
                st.rerun()
            except Exception as exc:
                log.error("triage.label.delete_failed", error=str(exc))
                st.error(f"Could not delete label: {exc}")


# ---- Main ------------------------------------------------------------------


def main() -> None:
    st.title("Triage")
    st.caption(
        "Review job titles found in your searches and mark them as relevant or not."
    )

    # Step 1 — Profile selector
    profiles = _load_profiles_safely()
    if not profiles:
        st.warning(
            "No profiles found. Create a profile on the Search config page first."
        )
        return

    profile_ids = [p.profile_id for p in profiles]
    profile_labels = {p.profile_id: p.name for p in profiles}

    active = st.session_state.get(_SESSION_KEY_ACTIVE_PROFILE)
    default_idx = profile_ids.index(active) if active in profile_ids else 0

    profile_id: str = st.selectbox(
        "Profile",
        options=profile_ids,
        index=default_idx,
        format_func=lambda pid: profile_labels.get(pid, pid),
        key="triage_profile_select",
    )

    st.divider()

    # Step 2 — Summary metrics
    _render_metrics(profile_id)

    st.divider()

    # Step 3 — Candidate table with inline labeling
    show_all: bool = st.checkbox(
        "Show already-labelled titles",
        value=False,
        key="triage_show_all",
    )

    candidates = _load_candidates(profile_id, include_decided=show_all)
    editor_df = _build_editor_df(candidates)

    # Step 4 — Batch actions (rendered before editor so state updates render)
    st.caption("Batch actions:")
    updated_df = _render_batch_actions(editor_df)

    edited = _render_candidate_editor(updated_df, show_all=show_all)

    st.divider()

    # Step 5 — Save + Apply
    _render_save_apply(edited, candidates, profile_id)

    # Step 6 — Label dictionary
    st.divider()
    _render_label_dictionary(profile_id)


main()
