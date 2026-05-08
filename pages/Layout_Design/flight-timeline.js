      });
    } else {
      state.flights = [];
    }
    if (typeof recomputeDuplicateApronByStandId === 'function') recomputeDuplicateApronByStandId();
    if (obj.simPlaybackPositionsByFlightId && typeof obj.simPlaybackPositionsByFlightId === 'object') {
      state.simPlaybackPositionsByFlightId = obj.simPlaybackPositionsByFlightId;
      state.simPlaybackTimelinesEvictedForMemory = false;
    }
    if (Object.prototype.hasOwnProperty.call(obj, '_airsideSimApply')) delete obj._airsideSimApply;
    state.simPlaying = false;
    state.layoutPathDrawPointer = null;
    if (typeof refreshHasSimulationResultFromPlaybackSources === 'function') {
      refreshHasSimulationResultFromPlaybackSources();
    } else {
      let playbackFlightCount = 0;
      (state.flights || []).forEach(function(f) {
        if (f && f.timeline && f.timeline.length >= 2) playbackFlightCount++;
      });
      state.hasSimulationResult = playbackFlightCount > 0;
    }
    const savedDlp = obj.simDeadlockGhostPlayback;
    let savedRc = 0;
    if (savedDlp && isFinite(Number(savedDlp.resolveCount))) savedRc = Math.floor(Number(savedDlp.resolveCount));
    if (state.simPlaybackPositionsByFlightId && typeof state.simPlaybackPositionsByFlightId === 'object') {
      state.simDeadlockGhostPlayback = deriveDeadlockGhostPlaybackFromPayload(
        { positions: state.simPlaybackPositionsByFlightId, deadlock_resolve_event_count: savedRc },
        state.flights
      );
    } else if (savedDlp && typeof savedDlp === 'object' && Array.isArray(savedDlp.events) && savedDlp.events.length) {
      state.simDeadlockGhostPlayback = {
        events: savedDlp.events.map(function(ev) {
          const o = { t_abs: Number(ev.t_abs), labels: Array.isArray(ev.labels) ? ev.labels.slice() : [] };
          if (ev.focusWorldX != null && isFinite(Number(ev.focusWorldX))) o.focusWorldX = Number(ev.focusWorldX);
          if (ev.focusWorldY != null && isFinite(Number(ev.focusWorldY))) o.focusWorldY = Number(ev.focusWorldY);
          return o;
        }),
        bodyLines: typeof savedDlp.bodyLines === 'string' ? savedDlp.bodyLines : '',
        resolveCount: savedRc,
      };
    } else {
      state.simDeadlockGhostPlayback = { events: [], bodyLines: '', resolveCount: 0 };
    }
    if (dp && dp.v === 1 && dp.simPlaybackEndCapSec != null && isFinite(Number(dp.simPlaybackEndCapSec))) {
      state.simPlaybackEndCapSec = Number(dp.simPlaybackEndCapSec);
    }
    state._pendingPersistSimWindow = null;
    if (dp && dp.v === 1
        && dp.simWindowStartSec != null && dp.simWindowEndSec != null
        && isFinite(Number(dp.simWindowStartSec)) && isFinite(Number(dp.simWindowEndSec))) {
      state._pendingPersistSimWindow = {
        lo: Number(dp.simWindowStartSec),
        hi: Number(dp.simWindowEndSec),
      };
    }
    applyDesignerPersistMapTypeAfterLoad(dp);
    syncMapTypePopoverFromState();
    if (typeof syncSimulationPlaybackAfterTimelines === 'function') syncSimulationPlaybackAfterTimelines();
    else if (typeof recomputeSimDuration === 'function') recomputeSimDuration();
    if (typeof redrawLayoutAfterEdit === 'function') redrawLayoutAfterEdit();
    else draw();
    let didAutoPathGraphSync = false;
    if (PATH_GRAPH_SYNC_ONLY_ON_EXPLICIT_ACTION && state.layers && state.layers.junction) {
      try {
        if (typeof applyPathGraphSyncNow === 'function') applyPathGraphSyncNow();
        didAutoPathGraphSync = true;
        if (typeof draw === 'function') draw();
        if (typeof markDesignerPageUpdateFresh === 'function') markDesignerPageUpdateFresh();
      } catch (ePg) {
        console.warn('applyLayoutObject: path graph sync', ePg);
      }
    }
    if (dp && dp.v === 1) {
      state.globalUpdateFresh = !!dp.globalUpdateFresh;
      if (!didAutoPathGraphSync) {
        if (dp.designerPageUpdateFresh === true) {
          if (typeof markDesignerPageUpdateFresh === 'function') markDesignerPageUpdateFresh();
        } else if (typeof markDesignerPageUpdateStale === 'function') {
          markDesignerPageUpdateStale();
        }
      }
      if (typeof syncProSimButtonFromDesignerPageState === 'function') syncProSimButtonFromDesignerPageState();
    }
    if (typeof renderFlightList === 'function') renderFlightList(false, false);
    if (typeof syncProSimButtonFromDesignerPageState === 'function') syncProSimButtonFromDesignerPageState();
  }
  /** E-series minutes; ARR_ROT_SEC / VTT_* / DTT_* / DEP_ROT_SEC from ``airside_sim`` schedule. */
  function applyAirsideScheduleRowToFlight(f, srec) {
    if (!f) return;
    if (!srec || typeof srec !== 'object') {
      delete f.eldtMin;
      delete f.eibtMin;
      delete f.eobtMin;
      delete f.etotMin;
      delete f.eldtMin;
      delete f.eibtMin;
      delete f.eobtMin;
      delete f.etotMin;
      delete f.eibtMinList;
      delete f.eobtMinList;
      delete f.ePushFinishedMinList;
      f.arrRotSec = null;
      delete f.proSimVttArrSec;
      delete f.proSimVttDepSec;
      delete f.proSimDttArrSec;
      delete f.proSimPushbackSec;
      delete f.proSimDttDepSec;
      delete f.proSimDepLineupSec;
      return;
    }
    function secOpt(key) {
      if (srec[key] == null || srec[key] === '') return NaN;
      const n = Number(srec[key]);
      return isFinite(n) ? n : NaN;
    }
    const eldtS = secOpt('ELDT');
    const eibtS = secOpt('EIBT');
    const eobtS = secOpt('EOBT');
    const etotS = secOpt('ETOT');
    if (isFinite(eldtS)) f.eldtMin = eldtS / 60;
    else delete f.eldtMin;
    if (isFinite(eibtS)) f.eibtMin = eibtS / 60;
    else delete f.eibtMin;
    if (isFinite(eobtS)) f.eobtMin = eobtS / 60;
    else delete f.eobtMin;
    if (isFinite(etotS)) f.etotMin = etotS / 60;
    else delete f.etotMin;
    function secListToMinList(key) {
      const raw = srec[key];
      if (!Array.isArray(raw)) return null;
      const out = raw.map(function(v) {
        const n = Number(v);
        return isFinite(n) ? n / 60 : null;
      });
      return out.length ? out : null;
    }
    const eibtList = secListToMinList('EIBT_LIST');
    const eobtList = secListToMinList('EOBT_LIST');
    const epushList = secListToMinList('E_PUSH_FINISHED_LIST');
    if (eibtList) f.eibtMinList = eibtList;
    else delete f.eibtMinList;
    if (eobtList) f.eobtMinList = eobtList;
    else delete f.eobtMinList;
    if (epushList) f.ePushFinishedMinList = epushList;
    else delete f.ePushFinishedMinList;
    const rotS = secOpt('ARR_ROT_SEC');
    if (isFinite(rotS)) f.arrRotSec = rotS;
    else f.arrRotSec = null;
    const vttArrS = secOpt('VTT_ARR_SEC');
    if (isFinite(vttArrS)) f.proSimVttArrSec = vttArrS;
    else delete f.proSimVttArrSec;
    const pushbackS = secOpt('PUSHBACK_SEC');
    if (isFinite(pushbackS)) f.proSimPushbackSec = pushbackS;
    else delete f.proSimPushbackSec;
    const vttDepS = secOpt('VTT_DEP_SEC');
    if (isFinite(vttDepS)) f.proSimVttDepSec = vttDepS;
    else delete f.proSimVttDepSec;
    const dttArrS = secOpt('DTT_ARR_SEC');
    if (isFinite(dttArrS)) f.proSimDttArrSec = dttArrS;
    else delete f.proSimDttArrSec;
    const dttDepS = secOpt('DTT_DEP_SEC');
    if (isFinite(dttDepS)) f.proSimDttDepSec = dttDepS;
    else delete f.proSimDttDepSec;
    const depRotS = secOpt('DEP_ROT_SEC');
    if (isFinite(depRotS)) f.proSimDepLineupSec = depRotS;
    else delete f.proSimDepLineupSec;
  }
  function flightEMinListForSchedule(f, listKey, metaSecListKey, scalarKey) {
    const raw = f && Array.isArray(f[listKey]) ? f[listKey] : null;
    if (raw && raw.length) return raw.map(function(v) {
      const n = Number(v);
      return isFinite(n) ? n : null;
    });
    const meta = f && f.timeline_meta && typeof f.timeline_meta === 'object' ? f.timeline_meta : null;
    const mraw = meta && Array.isArray(meta[metaSecListKey]) ? meta[metaSecListKey] : null;
    if (mraw && mraw.length) return mraw.map(function(v) {
      const n = Number(v);
      return isFinite(n) ? n / 60 : null;
    });
    const scalar = f && f[scalarKey] != null ? Number(f[scalarKey]) : NaN;
    return isFinite(scalar) ? [scalar] : [];
  }
  function isCompactPlaybackTrack(raw) {
    return !!(
      raw && typeof raw === 'object' && raw.format === 'compact_v2' &&
      Array.isArray(raw.t) && Array.isArray(raw.x) && Array.isArray(raw.y) && Array.isArray(raw.v) &&
      raw.t.length === raw.x.length && raw.t.length === raw.y.length && raw.t.length === raw.v.length
    );
  }
  function compactPlaybackTrackLength(raw) {
    return isCompactPlaybackTrack(raw) ? raw.t.length : 0;
  }
  function compactPlaybackTrackForFlight(f) {
    if (!f || f.id == null) return null;
    const map = state.simPlaybackPositionsByFlightId;
    if (!map || typeof map !== 'object') return null;
    const tr = map[String(f.id)];
    return isCompactPlaybackTrack(tr) ? tr : null;
  }
  function compactPlaybackDghostIntsMergedRangesSec(arr) {
    if (!Array.isArray(arr) || !arr.length) return [];
    const nums = [];
    for (let i = 0; i < arr.length; i++) {
      const v = Math.round(Number(arr[i]));
      if (isFinite(v)) nums.push(v);
    }
    if (!nums.length) return [];
    nums.sort(function(a, b) { return a - b; });
    const out = [];
    let s0 = nums[0], e0 = nums[0];
    for (let j = 1; j < nums.length; j++) {
      const n = nums[j];
      if (n <= e0 + 1) e0 = n;
      else {
        out.push([s0, e0]);
        s0 = e0 = n;
      }
    }
    out.push([s0, e0]);
    return out;
  }
  /** Starts of deadlock-ghost bursts: first tick of each contiguous group separated by gap > gapSec (deadlock banner uses 120). */
  function ghostSessionStartTimestampsSec(arr, gapSec) {
    const g = Math.max(30, Number(gapSec) || 120);
    const nums = [];
    for (let i = 0; i < arr.length; i++) {
      const v = Math.round(Number(arr[i]));
      if (isFinite(v)) nums.push(v);
    }
    if (!nums.length) return [];
    nums.sort(function(a, b) { return a - b; });
    const out = [nums[0]];
    for (let j = 1; j < nums.length; j++) {
      if (nums[j] - nums[j - 1] > g) out.push(nums[j]);
    }
    return out;
  }
  function compactPlaybackDghostSet(track) {
    const s = new Set();
    function addArr(a) {
      if (!Array.isArray(a)) return;
      for (let i = 0; i < a.length; i++) {
        const t = Math.round(Number(a[i]));
        if (isFinite(t)) s.add(t);
      }
    }
    if (track) {
      addArr(track.dghost_t);
    }
    return s;
  }
  function compactPlaybackDghostMergedRangesSec(track) {
    if (!track) return [];
    return compactPlaybackDghostIntsMergedRangesSec(track.dghost_t);
  }
  /** True if compact playback track records any deadlock ghost sample (simulation seconds). */
  function allocFlightTrackHasDeadlock(trDead) {
    return !!(
      trDead &&
      Array.isArray(trDead.dghost_t) &&
      trDead.dghost_t.length > 0
    );
  }
  /** Flight Schedule row: deadlock from last Pro Sim compact playback, timeline ghost, or persisted id set. */
  function flightScheduleRowHasDeadlock(f) {
    if (!f || f.id == null) return false;
    const idStr = String(f.id);
    if (state.deadlockFlightIdsFromLastSim && state.deadlockFlightIdsFromLastSim[idStr]) return true;
    const tr = compactPlaybackTrackForFlight(f);
    if (allocFlightTrackHasDeadlock(tr)) return true;
    if (f.timeline && Array.isArray(f.timeline)) {
      for (let i = 0; i < f.timeline.length; i++) {
        if (f.timeline[i] && f.timeline[i].deadlockGhost === true) return true;
      }
    }
    return false;
  }
  /** Gantt apron bar: timeline ghost overlay (time axis = sec/60). */
  function allocFlightDeadlockOverlayHtml(trDead, segT0Min, segT1Min, visT0Min, visT1Min) {
    const rN = trDead ? compactPlaybackDghostIntsMergedRangesSec(trDead.dghost_t) : [];
    if (!rN.length) return '';
    const denom = visT1Min - visT0Min;
    if (!(denom > 1e-12)) return '';
    function segmentsForRanges(ranges, cls) {
      const parts = [];
      for (let r = 0; r < ranges.length; r++) {
        const ds = ranges[r][0], de = ranges[r][1];
        const m0 = ds / 60;
        const m1 = (de + 1) / 60;
        const a = Math.max(visT0Min, m0, segT0Min);
        const b = Math.min(visT1Min, m1, segT1Min);
        if (!(b > a + 1e-12)) continue;
        const leftRel = ((a - visT0Min) / denom) * 100;
        const wRel = Math.max(0.5, ((b - a) / denom) * 100);
        parts.push(
          '<div class="' +
            cls +
            '" style="left:' +
            leftRel +
            '%;width:' +
            wRel +
            '%;"></div>'
        );
      }
      return parts;
    }
    const outHtml = segmentsForRanges(rN, 'alloc-flight-deadlock-seg');
    return outHtml.length ? outHtml.join('') : '';
  }
  function compactPlaybackTugIntervals(track) {
    if (!track) return [];
    if (Array.isArray(track.__tugIntervals)) return track.__tugIntervals;
    const raw = Array.isArray(track.tug_intervals) ? track.tug_intervals : [];
    const out = [];
    for (let i = 0; i < raw.length; i++) {
      const it = raw[i];
      if (!it || typeof it !== 'object') continue;
      const start = Number(it.start != null ? it.start : it.t0);
      const end = Number(it.end != null ? it.end : it.t1);
      if (isFinite(start) && isFinite(end) && end > start) out.push({ start: start, end: end });
    }
    track.__tugIntervals = out;
    return out;
  }
  function compactPlaybackNeedsTugAt(track, tSec) {
    const t = Number(tSec);
    if (!isFinite(t)) return false;
    const intervals = compactPlaybackTugIntervals(track);
    for (let i = 0; i < intervals.length; i++) {
      const it = intervals[i];
      if (t + 1e-9 >= it.start && t <= it.end + 1e-9) return true;
    }
    return false;
  }
  function compactPlaybackMetaStateAt(track, t) {
    const meta = Array.isArray(track && track.meta) ? track.meta : [];
    const out = {};
    const tt = Number(t);
    for (let i = 0; i < meta.length; i++) {
      const mr = meta[i];
      if (!mr || typeof mr !== 'object') continue;
      const mt = Number(mr.t);
      if (!isFinite(mt) || mt > tt + 1e-9) break;
      if (mr.phase != null) out.phase = String(mr.phase);
      if (mr.pathType != null) out.pathType = String(mr.pathType);
      if (mr.edgeId != null && String(mr.edgeId).trim()) out.edgeId = String(mr.edgeId).trim();
    }
    return out;
  }
  /**
   * Maps compact_v2 playback ``phase`` (airside_sim: Dep_taxi, Arr_taxi, Arr_taxi_occupied) to lookahead bump slot.
   * @returns {'dep'|'arr'|null}
   */
  function lookaheadMitigationBumpKindFromPlaybackPhase(phaseRaw) {
    const s = String(phaseRaw || '').trim().toLowerCase().replace(/-/g, '_').replace(/\s+/g, '_');
    if (s === 'dep_taxi') return 'dep';
    if (s === 'arr_taxi' || s === 'arr_taxi_occupied') return 'arr';
    return null;
  }
  /**
   * For each flight with dghost timestamps: per ghost-session start, +6 lookaheadDep if phase is Dep_taxi,
   * +6 lookaheadArr if Arr_taxi or Arr_taxi_occupied. Clamped 0–200 after ensureFlightLookaheadArrDepFlight.
   * @returns {number} number of flights with at least one field changed
   */
  function applyDeadlockMitigationLookaheadFromPlaybackTracks() {
    const positions = state.simPlaybackPositionsByFlightId;
    if (!positions || typeof positions !== 'object') return 0;
    let nFlights = 0;
    (state.flights || []).forEach(function(f) {
      if (!f || f.id == null) return;
      const raw = positions[String(f.id)];
      if (!isCompactPlaybackTrack(raw) || !Array.isArray(raw.dghost_t) || !raw.dghost_t.length) return;
      const starts = ghostSessionStartTimestampsSec(raw.dghost_t, 120);
      let addA = 0;
      let addD = 0;
      for (let si = 0; si < starts.length; si++) {
        const tr = starts[si];
        const m = compactPlaybackMetaStateAt(raw, tr);
        const kind = lookaheadMitigationBumpKindFromPlaybackPhase(m.phase);
        if (kind === 'dep') addD += 6;
        else if (kind === 'arr') addA += 6;
      }
      if (addA === 0 && addD === 0) return;
      ensureFlightLookaheadArrDepFlight(f);
      let touched = false;
      if (addD > 0) {
        f.lookaheadDep = Math.max(0, Math.min(200, Math.floor(Number(f.lookaheadDep) || 0) + addD));
        touched = true;
      }
      if (addA > 0) {
        f.lookaheadArr = Math.max(0, Math.min(200, Math.floor(Number(f.lookaheadArr) || 0) + addA));
        touched = true;
      }
      if (touched) nFlights++;
    });
    return nFlights;
  }
  function compactPlaybackSampleAtIndex(track, idx) {
    if (!isCompactPlaybackTrack(track)) return null;
    const i = Math.max(0, Math.min(track.t.length - 1, idx | 0));
    const t = Number(track.t[i]);
    const x = Number(track.x[i]);
    const y = Number(track.y[i]);
    if (!isFinite(t) || !isFinite(x) || !isFinite(y)) return null;
    const o = { t: t, x: x, y: y, v: Number(track.v[i]) || 0 };
    const trR = Math.round(Number(t));
    o.deadlockGhost = compactPlaybackDghostSet(track).has(trR);

    const m = compactPlaybackMetaStateAt(track, t);
    if (m.phase) o.phase = m.phase;
    if (m.pathType) o.pathType = m.pathType;
    if (m.edgeId) o.edgeId = m.edgeId;
    return o;
  }
  function compactPlaybackIndexAtTime(track, tSec, clampEnd) {
    if (!isCompactPlaybackTrack(track) || track.t.length < 2) return -1;
    let t = Number(tSec);
    if (!isFinite(t)) return -1;
    const firstT = Number(track.t[0]);
    const lastT = Number(track.t[track.t.length - 1]);
    if (t + 1e-9 < firstT) return -1;
    if (t > lastT) {
      if (!clampEnd) return -1;
      t = lastT;
    }
    let lo = 0, hi = track.t.length - 1;
    while (lo < hi) {
      const mid = Math.ceil((lo + hi) / 2);
      if (Number(track.t[mid]) <= t + 1e-9) lo = mid;
      else hi = mid - 1;
    }
    const idx = Math.min(lo, track.t.length - 2);
    return (t + 1e-9 >= Number(track.t[idx]) && t - 1e-9 <= Number(track.t[idx + 1])) ? idx : -1;
  }
  /** Last unit direction of a non-trivial chord strictly before the segment containing ``tSec`` (full track — not windowed). */
  function playbackLastMotionUnitDirBeforeTime(track, tSec, skipPushbackOpt) {
    const skipPushback = skipPushbackOpt === true;
    const eps = 0.08;
    const eps2 = eps * eps;
    if (!isCompactPlaybackTrack(track)) return null;
    const idx = compactPlaybackIndexAtTime(track, tSec, true);
    if (idx < 1) return null;
    for (let j = idx - 1; j >= 0; j--) {
      const p = compactPlaybackSampleAtIndex(track, j);
      const q = compactPlaybackSampleAtIndex(track, j + 1);
      if (!p || !q) continue;
      if (skipPushback && String(p.phase || '') === 'Pushback') continue;
      const ddx = q.x - p.x, ddy = q.y - p.y;
      const l2 = ddx * ddx + ddy * ddy;
      if (l2 >= eps2) {
        const inv = 1 / Math.sqrt(l2);
        return { dx: ddx * inv, dy: ddy * inv };
      }
    }
    return null;
  }
  function compactPlaybackXYAtAbsTime(track, tSec) {
    const idx = compactPlaybackIndexAtTime(track, tSec, true);
    if (idx < 0 || !isCompactPlaybackTrack(track)) return null;
    const t = Number(tSec);
    const t0 = Number(track.t[idx]), t1 = Number(track.t[idx + 1]);
    const x0 = Number(track.x[idx]), y0 = Number(track.y[idx]);
    const x1 = Number(track.x[idx + 1]), y1 = Number(track.y[idx + 1]);
    if (!isFinite(t0) || !isFinite(t1) || !isFinite(x0) || !isFinite(y0) || !isFinite(x1) || !isFinite(y1)) return null;
    if (t1 <= t0) return { x: x0, y: y0 };
    const u = Math.max(0, Math.min(1, (t - t0) / (t1 - t0)));
    return { x: x0 + (x1 - x0) * u, y: y0 + (y1 - y0) * u };
  }
  function deadlockFocusWorldMeanAtRoundedTime(positions, flights, tR) {
    const tr = Math.round(Number(tR));
    if (!isFinite(tr) || !positions || typeof positions !== 'object') return null;
    let sx = 0, sy = 0, n = 0;
    (flights || []).forEach(function(f) {
      if (!f || f.id == null) return;
      const raw = positions[String(f.id)];
      if (!isCompactPlaybackTrack(raw)) return;
      if (!compactPlaybackDghostSet(raw).has(tr)) return;
      const p = compactPlaybackXYAtAbsTime(raw, tr);
      if (p && isFinite(p.x) && isFinite(p.y)) {
        sx += p.x;
        sy += p.y;
        n++;
      }
    });
    if (!n) return null;
    return { x: sx / n, y: sy / n };
  }
  function focusLayoutMapOnWorldXY(wx, wy) {
    if (!isFinite(wx) || !isFinite(wy)) return;
    const c = document.getElementById('grid-canvas');
    if (!c) return;
    const w = c.clientWidth || 0, h = c.clientHeight || 0;
    if (w < 8 || h < 8) return;
    const sc = Math.max(Number(state.scale) || 1, 1e-6);
    state.panX = w * 0.5 - wx * sc;
    state.panY = h * 0.5 - wy * sc;
    if (typeof scheduleDraw === 'function') scheduleDraw();
    else if (typeof draw === 'function') {
      try { draw(); } catch (e) { /* ignore */ }
    }
  }
  function seekSimToDeadlockMarkerEvent(ev) {
    if (!ev || typeof ev !== 'object') return;
    const tAbs = Number(ev.t_abs);
    if (!isFinite(tAbs)) return;
    const lo = Number(state.simStartSec), hi = Number(state.simDurationSec);
    if (!isFinite(lo) || !isFinite(hi)) return;
    const snapped =
      typeof snapSimTimeToPlaybackWindowSec === 'function'
        ? snapSimTimeToPlaybackWindowSec(tAbs)
        : (typeof snapSimTimeSecForSlider === 'function'
          ? snapSimTimeSecForSlider(Math.max(lo, Math.min(hi, tAbs)))
          : Math.max(lo, Math.min(hi, tAbs)));
    state.simTimeSec = snapped;
    const slider = document.getElementById('flightSimSlider');
    if (slider) slider.value = String(snapped);
    if (typeof updateFlightSimPlaybackLabelsDom === 'function') updateFlightSimPlaybackLabelsDom();
    let fx = ev.focusWorldX != null ? Number(ev.focusWorldX) : NaN;
    let fy = ev.focusWorldY != null ? Number(ev.focusWorldY) : NaN;
    if (!isFinite(fx) || !isFinite(fy)) {
      const fw = deadlockFocusWorldMeanAtRoundedTime(
        state.simPlaybackPositionsByFlightId,
        state.flights,
        Math.round(snapped)
      );
      if (fw) {
        fx = fw.x;
        fy = fw.y;
      }
    }
    if (isFinite(fx) && isFinite(fy)) focusLayoutMapOnWorldXY(fx, fy);
    try {
      if (typeof draw === 'function') draw();
    } catch (e2) { /* ignore */ }
    if (typeof update3DSceneWhenVisible === 'function') update3DSceneWhenVisible();
  }
  function compactPlaybackTimelineWindow(track, tSec, radius) {
    const idx = compactPlaybackIndexAtTime(track, tSec, true);
    if (idx < 0) return null;
    const r = Math.max(2, radius || 40);
    const s = Math.max(0, idx - r);
    const e = Math.min(track.t.length - 1, idx + r + 1);
    const out = [];
    for (let i = s; i <= e; i++) {
      const p = compactPlaybackSampleAtIndex(track, i);
      if (p) out.push(p);
    }
    return out.length >= 2 ? out : null;
  }
  function compactPlaybackTrackStartEnd(track) {
    if (!isCompactPlaybackTrack(track) || track.t.length < 1) return null;
    return { t0: Number(track.t[0]), t1: Number(track.t[track.t.length - 1]) };
  }
  function refreshHasSimulationResultFromPlaybackSources() {
    const pos = state.simPlaybackPositionsByFlightId;
    if (pos && typeof pos === 'object') {
      for (const k in pos) {
        if (!Object.prototype.hasOwnProperty.call(pos, k)) continue;
        if (compactPlaybackTrackLength(pos[k]) >= 2) {
          state.hasSimulationResult = true;
          return;
        }
      }
    }
    state.hasSimulationResult = false;
    state.deadlockMitigateBannerRerunHint = false;
    state.simDeadlockGhostPlayback = { events: [], bodyLines: '', resolveCount: 0 };
  }
  function evictFlightPlaybackTimelinesWhenPlayBlocked() {
    if (!state.simPlaybackPositionsByFlightId || typeof state.simPlaybackPositionsByFlightId !== 'object') return false;
    const flights = state.flights || [];
    for (let i = 0; i < flights.length; i++) {
      const f = flights[i];
      if (!f) continue;
      f.timeline = null;
    }
    state.simPlaybackTimelinesEvictedForMemory = true;
    _lazyTimelineLastEvictSimSec = NaN;
    refreshHasSimulationResultFromPlaybackSources();
    return true;
  }
  function rehydrateFlightPlaybackTimelinesAfterPlayAllowed() {
    if (!state.simPlaybackTimelinesEvictedForMemory) return false;
    refreshHasSimulationResultFromPlaybackSources();
    state.simPlaybackTimelinesEvictedForMemory = false;
    return true;
  }
  function deriveDeadlockGhostPlaybackFromPayload(payload, flights) {
    const empty = { events: [], bodyLines: '', resolveCount: 0 };
    if (!payload || typeof payload !== 'object') return empty;
    const positions = payload.positions;
    if (!positions || typeof positions !== 'object') return empty;
    const resolveCount = Number(payload.deadlock_resolve_event_count);
    const rc = isFinite(resolveCount) && resolveCount > 0 ? Math.floor(resolveCount) : 0;
    const byT = new Map();
    function ingestTrackArray(raw, f, arr) {
      if (!Array.isArray(arr) || !arr.length) return;
      const starts = ghostSessionStartTimestampsSec(arr, 120);
      for (let si = 0; si < starts.length; si++) {
        const tr = starts[si];
        const label =
          String(f.reg || '').trim() ||
          String(f.flightNumber || f.id || '').trim() ||
          String(f.id);
        if (!byT.has(tr))
          byT.set(tr, { labels: [] });
        const bx = byT.get(tr);
        if (bx.labels.indexOf(label) < 0) bx.labels.push(label);
      }
    }
    (flights || []).forEach(function(f) {
      if (!f || f.id == null) return;
      const raw = positions[String(f.id)];
      if (!isCompactPlaybackTrack(raw)) return;
      ingestTrackArray(raw, f, raw.dghost_t);
    });
    const entries = Array.from(byT.entries()).sort(function(a, b) { return a[0] - b[0]; });
    const events = entries.map(function(e) {
      const tR = e[0];
      const bx = e[1];
      const ev = { t_abs: tR, labels: bx.labels.slice() };
      const fw = deadlockFocusWorldMeanAtRoundedTime(positions, flights, tR);
      if (fw && isFinite(fw.x) && isFinite(fw.y)) {
        ev.focusWorldX = fw.x;
        ev.focusWorldY = fw.y;
      }
      return ev;
    });
    let bodyLines = '';
    if (events.length) {
      const chunks = [];
      for (let ei = 0; ei < events.length; ei++) {
        const ev = events[ei];
        const timeStr = formatTotalSecondsToHHMMSS(ev.t_abs);
        const lbls = (ev.labels && ev.labels.length) ? ev.labels : [];
        let pushedAny = false;
        for (let li = 0; li < lbls.length; li++) {
          const reg = String(lbls[li] || '').trim();
          if (!reg) continue;
          chunks.push(timeStr + '  ' + reg);
          pushedAny = true;
        }
        if (!pushedAny) chunks.push(timeStr + '  —');
      }
      bodyLines = chunks.join('\n');
    } else if (rc > 0) {
      bodyLines =
        'Resolves: ' + rc + '  (no ghost ticks in positions)';
    }
    return {
      events: events,
      bodyLines: bodyLines,
      resolveCount: rc,
    };
  }
  function renderFlightSimSliderDeadlockMarkers() {
    const host = document.getElementById('flightSimSliderMarkers');
    if (!host) return;
    host.textContent = '';
    const span = Number(state.simDurationSec) - Number(state.simStartSec);
    if (!(span > 1e-6)) return;
    const dlp = state.simDeadlockGhostPlayback;
    const evs = (dlp && Array.isArray(dlp.events)) ? dlp.events : [];
    const lo = Number(state.simStartSec);
    const hi = Number(state.simDurationSec);
    evs.forEach(function(ev) {
      const t = Number(ev.t_abs);
      if (!isFinite(t)) return;
      if (t < lo - 2 || t > hi + 2) return;
      const pct = 100 * (t - lo) / span;
      const dot = document.createElement('span');
      dot.className = 'sim-slider-deadlock-dot';
      dot.style.left = Math.max(0, Math.min(100, pct)) + '%';
      dot.setAttribute(
        'title',
        'Deadlock @ ' +
          formatTotalSecondsToHHMMSS(t) +
          (ev.labels && ev.labels.length
            ? ' — ' + ev.labels.map(function(u) {
              return String(u || '').trim();
            }).filter(Boolean).join(', ')
            : '') +
          ' — click to jump'
      );
      dot.setAttribute('role', 'button');
      dot.setAttribute('tabindex', '0');
      dot.addEventListener('click', function(e) {
        e.preventDefault();
        e.stopPropagation();
        seekSimToDeadlockMarkerEvent(ev);
      });
      dot.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          e.stopPropagation();
          seekSimToDeadlockMarkerEvent(ev);
        }
      });
      host.appendChild(dot);
    });
  }
  function applyAirsideSimulationResultPayload(payload) {
    if (!payload || typeof payload !== 'object') return;
    state.deadlockMitigateBannerRerunHint = false;
    const truncCap = (payload.simulation_truncated_deadlock === true || payload.simulation_truncated_stot_horizon === true)
      ? (function() {
        const rawCap = payload.simulation_playback_end_abs_sec;
        const c = Number(rawCap);
        return isFinite(c) ? c : null;
      })()
      : null;
    const flightsDetail = Array.isArray(payload.flights_detail) ? payload.flights_detail : null;
    if (flightsDetail) {
      const byId = {};
      flightsDetail.forEach(function(row) {
        if (!row || row.flight_id == null) return;
        const fid = String(row.flight_id);
        const fin = row.edge_list_finished;
        const planned = row.edge_list;
        if (Array.isArray(fin) && fin.length) {
          byId[fid] = fin.slice();
        } else if (Array.isArray(planned) && planned.length) {
          byId[fid] = planned.slice();
        } else {
          byId[fid] = [];
        }
      });
      (state.flights || []).forEach(function(f) {
        if (!f || f.id == null) return;
        const raw = byId[String(f.id)];
        if (Array.isArray(raw) && raw.length) {
          f.edge_list = raw.slice();
          f.proSimEdgeList = f.edge_list.slice();
        } else {
          delete f.edge_list;
          delete f.proSimEdgeList;
        }
      });
    }
    const positions = payload.positions;
    const hasPositions = positions && typeof positions === 'object' && Object.keys(positions).length > 0;
    const scheduleList = Array.isArray(payload.schedule) ? payload.schedule : [];
    const layout = payload.layout;
    if (layout && typeof layout === 'object') {
      applyLayoutObject(layout);
    }
    state.simPlaybackEndCapSec = truncCap;
    const schedById = {};
    scheduleList.forEach(function(s) {
      if (s && s.flight_id != null) schedById[String(s.flight_id)] = s;
    });
    let mergedTimelines = 0;
    (state.flights || []).forEach(function(f) {
      if (!f || f.id == null) return;
      const srec = schedById[String(f.id)] || null;
      const track = hasPositions ? positions[String(f.id)] : null;
      const hasTrack = compactPlaybackTrackLength(track) >= 2;
      f.timeline = null;
      if (hasTrack) {
        mergedTimelines++;
        if (f.arrDep !== 'Dep') f.arrRetFailed = false;
      }
      if (srec && hasTrack) {
        const eldtS = srec.ELDT != null ? Number(srec.ELDT) : NaN;
        const eibtS = srec.EIBT != null ? Number(srec.EIBT) : NaN;
        const eobtS = srec.EOBT != null ? Number(srec.EOBT) : NaN;
        const etotS = srec.ETOT != null ? Number(srec.ETOT) : NaN;
        const prevMeta = f.timeline_meta || {};
        const builtDep = (typeof buildDepartureSurfaceTimelineSegments === 'function' && f.arrDep === 'Dep'
          && isFinite(eobtS) && isFinite(etotS))
          ? buildDepartureSurfaceTimelineSegments(f, eobtS, etotS)
          : null;
        const builtDepMeta = (builtDep && builtDep.meta) ? builtDep.meta : null;
        f.timeline_meta = Object.assign(
          {},
          prevMeta,
          builtDepMeta || {},
          {
            playbackSource: 'des_result',
            eldtSec: isFinite(eldtS) ? eldtS : undefined,
            eibtSec: isFinite(eibtS) ? eibtS : undefined,
            eobtSec: isFinite(eobtS) ? eobtS : undefined,
            etotSec: isFinite(etotS) ? etotS : undefined,
            eibtSecList: Array.isArray(srec.EIBT_LIST) ? srec.EIBT_LIST.slice() : undefined,
            eobtSecList: Array.isArray(srec.EOBT_LIST) ? srec.EOBT_LIST.slice() : undefined,
            ePushFinishedSecList: Array.isArray(srec.E_PUSH_FINISHED_LIST) ? srec.E_PUSH_FINISHED_LIST.slice() : undefined,
          }
        );
      } else {
        delete f.timeline_meta;
      }
      applyAirsideScheduleRowToFlight(f, srec);
    });
    state.hasSimulationResult = mergedTimelines > 0;
    state.simPlaybackPositionsByFlightId = hasPositions ? positions : null;
    state.simPlaybackScheduleSnapshot = scheduleList.length ? scheduleList.slice() : null;
    state.simPlaybackTimelinesEvictedForMemory = false;
    state.simDeadlockGhostPlayback = deriveDeadlockGhostPlaybackFromPayload(payload, state.flights);
    if (hasPositions && positions && typeof positions === 'object') {
      if (!state.deadlockFlightIdsFromLastSim) state.deadlockFlightIdsFromLastSim = Object.create(null);
      Object.keys(positions).forEach(function(pid) {
        const tr = positions[pid];
        if (allocFlightTrackHasDeadlock(tr)) state.deadlockFlightIdsFromLastSim[String(pid)] = true;
      });
    }
    if (state.hasSimulationResult) {
      if (typeof markGlobalUpdateFresh === 'function') markGlobalUpdateFresh();
      if (typeof markDesignerPageUpdateFresh === 'function') markDesignerPageUpdateFresh();
    } else if (typeof markGlobalUpdateStale === 'function') markGlobalUpdateStale();
    if (typeof syncSimulationPlaybackAfterTimelines === 'function') syncSimulationPlaybackAfterTimelines();
    else if (typeof recomputeSimDuration === 'function') recomputeSimDuration();
    if (typeof resizeCanvas === 'function') resizeCanvas();
    if (typeof reset2DView === 'function') reset2DView();
    if (typeof syncPanelFromState === 'function') syncPanelFromState();
    if (typeof renderFlightList === 'function') renderFlightList(false, false);
    if (typeof renderKpiDashboard === 'function') renderKpiDashboard('Updated');
    if (typeof renderRunwaySeparation === 'function') renderRunwaySeparation();
    if (typeof draw === 'function') draw();
    if (typeof update3DSceneWhenVisible === 'function') update3DSceneWhenVisible();
    if (typeof syncProSimButtonFromDesignerPageState === 'function') syncProSimButtonFromDesignerPageState();
  }
  function applyInitialLayoutFromJson() {
    if (!INITIAL_LAYOUT || typeof INITIAL_LAYOUT !== 'object') return;
    applyLayoutObject(INITIAL_LAYOUT);
  }
  function updateLayoutNameBar(name) {
    const n = (name && String(name).trim()) || '';
    state.currentLayoutName = n || state.currentLayoutName || 'default_layout';
    const bar = document.getElementById('layout-name-bar');
    if (bar) bar.textContent = n || state.currentLayoutName;
  }
  function uniqueNameAgainstSet(baseName, usedNames) {
    const base = (baseName && String(baseName).trim()) || 'Untitled';
    const used = usedNames instanceof Set ? usedNames : new Set();
    if (!used.has(base)) return base;
    let idx = 1;
    while (used.has(base + ' (' + idx + ')')) idx++;
    return base + ' (' + idx + ')';
  }
  function zeroPadNumber(num, width) {
    return String(Math.max(0, Number(num) || 0)).padStart(width, '0');
  }
  function getDefaultPathName(pathType, currentId) {
    const prefix = pathType === 'runway' ? 'RW' : (pathType === 'runway_exit' ? 'RTX' : (pathType === 'apron_taxiway' ? 'ATX' : (pathType === 'general_queue_taxiway' ? 'QTX' : 'TX')));
    const sameType = (state.taxiways || []).filter(function(tw) { return tw && tw.id !== currentId && tw.pathType === pathType; });
    const used = new Set(sameType.map(function(tw) { return (tw.name && String(tw.name).trim()) || ''; }).filter(Boolean));
    let n = 1;
    let candidate = prefix + String(n);
    while (used.has(candidate)) {
      n++;
      candidate = prefix + String(n);
      if (n > 100000) break;
    }
    return candidate;
  }
  function getDefaultTerminalName(currentId) {
    return getDefaultBuildingNameForType(BUILDING_TYPE_DEFAULT, currentId);
  }
  function getDefaultPbbStandName(currentId) {
    const stands = (state.pbbStands || []).filter(function(st) { return st && st.id !== currentId; });
    const used = new Set(stands.map(function(st) { return (st.name && String(st.name).trim()) || ''; }).filter(Boolean));
    return uniqueNameAgainstSet('C' + zeroPadNumber(stands.length + 1, 3), used);
  }
  function getDefaultRemoteStandName(currentId) {
    const stands = (state.remoteStands || []).filter(function(st) { return st && st.id !== currentId; });
    const used = new Set(stands.map(function(st) { return (st.name && String(st.name).trim()) || ''; }).filter(Boolean));
    return uniqueNameAgainstSet('R' + zeroPadNumber(stands.length + 1, 3), used);
  }
  function getDefaultTempStandName(currentId) {
    const stands = (state.tempStands || []).filter(function(st) { return st && st.id !== currentId; });
    const used = new Set(stands.map(function(st) { return (st.name && String(st.name).trim()) || ''; }).filter(Boolean));
    return uniqueNameAgainstSet('T' + zeroPadNumber(stands.length + 1, 3), used);
  }
  function getApronLinkDefaultName(linkOrId) {
    const linkId = (typeof linkOrId === 'object' && linkOrId) ? linkOrId.id : linkOrId;
    const idx = (state.apronLinks || []).findIndex(function(lk) { return lk && lk.id === linkId; });
    return 'Leadin Taxiway ' + String(idx >= 0 ? idx + 1 : ((state.apronLinks || []).length + 1));
  }
  function getApronLinkDisplayName(link) {
    if (!link) return 'Leadin Taxiway';
    return (link.name && String(link.name).trim()) || getApronLinkDefaultName(link);
  }
  function ensureUniqueApronLinkName(rawName, currentId) {
    const fallbackBase = getApronLinkDefaultName(currentId);
    const baseName = (rawName && String(rawName).trim()) || fallbackBase;
    const used = new Set((state.apronLinks || [])
      .filter(function(lk) { return lk && lk.id !== currentId; })
      .map(function(lk) { return (lk.name && String(lk.name).trim()) || getApronLinkDefaultName(lk); })
      .filter(Boolean));
    return uniqueNameAgainstSet(baseName, used);
  }
  function getLayoutEdgeDefaultName(edge) {
    if (!edge) return 'Edge';
    return 'Edge ' + (edge.label || '001');
  }
  function getLayoutEdgeDisplayName(edge) {
    if (!edge) return 'Edge';
    return (edge.name && String(edge.name).trim()) || getLayoutEdgeDefaultName(edge);
  }
  function ensureUniqueLayoutEdgeName(rawName, currentId, fallbackEdge) {
    const fallbackBase = getLayoutEdgeDefaultName(fallbackEdge || { label: '001' });
    const baseName = (rawName && String(rawName).trim()) || fallbackBase;
    const used = new Set(Object.keys(state.layoutEdgeNames || {})
      .filter(function(id) { return id !== currentId; })
      .map(function(id) { return state.layoutEdgeNames[id]; })
      .filter(Boolean));
    return uniqueNameAgainstSet(baseName, used);
  }
  function normalizeLayoutNameKey(name) {
    return String(name || '').trim().toLowerCase();
  }
  function findDuplicateLayoutName(objectKind, excludeId, proposedRaw) {
    const key = normalizeLayoutNameKey(proposedRaw);
    if (!key) return null;
    const ex = excludeId == null || excludeId === '' ? null : String(excludeId);
    function isOther(oid) {
      if (ex === null) return true;
      return String(oid) !== ex;
    }
    if (objectKind === 'terminal') {
      const arr = state.terminals || [];
      for (let i = 0; i < arr.length; i++) {
        const o = arr[i];
        if (!o || !isOther(o.id)) continue;
        const disp = (o.name && String(o.name).trim()) || '';
        if (normalizeLayoutNameKey(disp) === key) return { kind: 'terminal', existing: disp || o.id };
      }
      return null;
    }
    if (objectKind === 'pbb') {
      const arr = state.pbbStands || [];
      for (let i = 0; i < arr.length; i++) {
        const o = arr[i];
        if (!o || !isOther(o.id)) continue;
        const disp = (o.name && String(o.name).trim()) || '';
        if (normalizeLayoutNameKey(disp) === key) return { kind: 'pbb', existing: disp || o.id };
      }
      return null;
    }
    if (objectKind === 'remote') {
      const arr = state.remoteStands || [];
      for (let i = 0; i < arr.length; i++) {
        const o = arr[i];
        if (!o || !isOther(o.id)) continue;
        const disp = (o.name && String(o.name).trim()) || '';
        if (normalizeLayoutNameKey(disp) === key) return { kind: 'remote', existing: disp || o.id };
      }
      return null;
    }
    if (objectKind === 'tempStand') {
      const arr = state.tempStands || [];
      for (let i = 0; i < arr.length; i++) {
        const o = arr[i];
        if (!o || !isOther(o.id)) continue;
        const disp = (o.name && String(o.name).trim()) || '';
        if (normalizeLayoutNameKey(disp) === key) return { kind: 'tempStand', existing: disp || o.id };
      }
      return null;
    }
    if (objectKind === 'holdingPoint') {
      const arr = state.holdingPoints || [];
      for (let i = 0; i < arr.length; i++) {
        const o = arr[i];
        if (!o || !isOther(o.id)) continue;
        const disp = (o.name && String(o.name).trim()) || '';
        if (normalizeLayoutNameKey(disp) === key) return { kind: 'holdingPoint', existing: disp || o.id };
      }
      return null;
    }
    if (objectKind === 'taxiway') {
      const arr = state.taxiways || [];
      for (let i = 0; i < arr.length; i++) {
        const o = arr[i];
        if (!o || !isOther(o.id)) continue;
        const disp = (o.name && String(o.name).trim()) || '';
        if (normalizeLayoutNameKey(disp) === key) return { kind: 'taxiway', existing: disp || o.id };
      }
      return null;
    }
    if (objectKind === 'apronLink') {
      const arr = state.apronLinks || [];
      for (let i = 0; i < arr.length; i++) {
        const o = arr[i];
        if (!o || !isOther(o.id)) continue;
        const disp = getApronLinkDisplayName(o);
        if (normalizeLayoutNameKey(disp) === key) return { kind: 'apronLink', existing: disp };
      }
      return null;
    }
    if (objectKind === 'layoutEdge') {
      const map = state.layoutEdgeNames || {};
      const edgeIds = Object.keys(map);
      for (let ki = 0; ki < edgeIds.length; ki++) {
        const kid = edgeIds[ki];
        if (!isOther(kid)) continue;
        const disp = map[kid];
        if (disp != null && normalizeLayoutNameKey(disp) === key) return { kind: 'layoutEdge', existing: String(disp) };
      }
      return null;
    }
    return null;
  }
  function alertDuplicateLayoutName() {
    alert('설정 불가: 동일한 이름이 이미 사용 중입니다.');
  }
  function ensureDefaultDirectionModes() {
    if (state.directionModes.length === 0) {
      state.directionModes = [
        { id: id(), name: 'Mode A', direction: 'clockwise' },
        { id: id(), name: 'Mode B', direction: 'counter_clockwise' },
        { id: id(), name: 'Mode C', direction: 'both' }
      ];
    }
  }
  const undoStack = [];
  const maxUndoLevels = _interactionConfigNum('maxUndoLevels', 50);
  function pushUndo() {
    const snap = {
      terminals: JSON.parse(JSON.stringify(state.terminals || [])),
      pbbStands: JSON.parse(JSON.stringify(state.pbbStands || [])),
      remoteStands: JSON.parse(JSON.stringify(state.remoteStands || [])),
      tempStands: JSON.parse(JSON.stringify(state.tempStands || [])),
      holdingPoints: JSON.parse(JSON.stringify(state.holdingPoints || [])),
      taxiways: JSON.parse(JSON.stringify(state.taxiways || [])),
      apronLinks: JSON.parse(JSON.stringify(state.apronLinks || [])),
      layoutImageOverlay: JSON.parse(JSON.stringify(state.layoutImageOverlay || null)),
      layoutEdgeNames: JSON.parse(JSON.stringify(state.layoutEdgeNames || {})),
      directionModes: JSON.parse(JSON.stringify(state.directionModes || [])),
      flights: cloneFlightsWithoutPathPolylineCache(state.flights),
      layoutMarkers: JSON.parse(JSON.stringify(state.layoutMarkers || []))
    };
    undoStack.push(snap);
    if (undoStack.length > maxUndoLevels) undoStack.shift();
    if (typeof markGlobalUpdateStale === 'function') markGlobalUpdateStale();
  }
  function undo() {
    if (!undoStack.length) return;
    const snap = undoStack.pop();
    state.terminals = snap.terminals;
    state.pbbStands = snap.pbbStands;
    state.remoteStands = snap.remoteStands;
    state.tempStands = snap.tempStands || [];
    state.holdingPoints = snap.holdingPoints || [];
    state.taxiways = snap.taxiways;
    state.apronLinks = snap.apronLinks;
    state.apronLinkJunctionOverlayDirtyIds = null;
    state.layoutImageOverlay = normalizeLayoutImageOverlay(snap.layoutImageOverlay);
    syncLayoutImageBitmap();
    state.layoutEdgeNames = snap.layoutEdgeNames || {};
    state.directionModes = snap.directionModes;
    state.flights = snap.flights;
    state.layoutMarkers = normalizeLayoutMarkerAreaZOrder(Array.isArray(snap.layoutMarkers) ? snap.layoutMarkers : []);
    state.pathArcDrag = null;
    state.selectedObject = null;
    state.currentTerminalId = state.terminals.length ? state.terminals[0].id : null;
    state.terminalDrawingId = null;
    state.taxiwayDrawingId = null;
    state.layoutPathDrawPointer = null;
    syncPanelFromState();
    updateObjectInfo();
    renderObjectList();
    if (typeof redrawLayoutAfterEdit === 'function') redrawLayoutAfterEdit();
    else if (typeof updateAllFlightPaths === 'function') updateAllFlightPaths(); else draw();
  }
  function getTaxiwayDirection(tw) {
    if (!tw) return 'both';
    if (tw.direction != null) {
      const d = tw.direction;
      if (d === 'topToBottom') return 'clockwise';
      if (d === 'bottomToTop') return 'counter_clockwise';
      return d || 'both';
    }
    if (tw.directionModeId) {
      const m = state.directionModes.find(d => d.id === tw.directionModeId);
      if (m && m.direction) return m.direction;
    }
    return 'both';
  }
  function normalizeRwDirectionValue(dir) {
    if (dir == null) return 'both';
    const s0 = String(dir).trim();
    if (!s0) return 'both';
    const s = s0.toLowerCase().replace(/[\s-]+/g, '_');
    if (s === 'clockwise' || s === 'cw') return 'clockwise';
    if (s === 'counter_clockwise' || s === 'ccw' || s === 'counterclockwise') return 'counter_clockwise';
    if (s === 'top_tobottom' || s === 'toptobottom' || s === 'ttb') return 'clockwise';
    if (s === 'bottom_totop' || s === 'bottomtotop' || s === 'btt') return 'counter_clockwise';
    return 'both';
  }
  function normalizeAllowedRunwayDirections(raw) {
    const out = [];
    const src = Array.isArray(raw) ? raw : [];
    src.forEach(function(v) {
      const d = normalizeRwDirectionValue(v);
      if (d === 'clockwise' && out.indexOf('clockwise') < 0) out.push('clockwise');
      if (d === 'counter_clockwise' && out.indexOf('counter_clockwise') < 0) out.push('counter_clockwise');
    });
    return out;
  }
  function getTaxiwayAllowedRunwayDirections(tw) {
    if (!tw || tw.pathType !== 'runway_exit') return (RW_EXIT_ALLOWED_DEFAULT && RW_EXIT_ALLOWED_DEFAULT.length) ? RW_EXIT_ALLOWED_DEFAULT.slice() : ['clockwise', 'counter_clockwise'];
    const arr = normalizeAllowedRunwayDirections(tw.allowedRwDirections);
    if (!arr.length) return (RW_EXIT_ALLOWED_DEFAULT && RW_EXIT_ALLOWED_DEFAULT.length) ? RW_EXIT_ALLOWED_DEFAULT.slice() : ['clockwise', 'counter_clockwise'];
    return arr;
  }
  function isRunwayExitDirectionAllowed(tw, runwayDir) {
    const d = normalizeRwDirectionValue(runwayDir);
    if (d !== 'clockwise' && d !== 'counter_clockwise') return true;
    const allow = getTaxiwayAllowedRunwayDirections(tw);
    return allow.indexOf(d) >= 0;
  }
  /**
   * Arrival RET sampling (F2): runways in the property panel use CW / CCW / Both; ``both`` keeps
   * vertex order and relies on Infra schedule (path ops) per time slot. When matching RET "Available
   * RW direction", we use CW as the operational stand-in for ambiguous ``both``.
   */
  function getRunwayOperationalDirForArrivalRetFilter2(rw) {
    if (!rw || rw.pathType !== 'runway') return 'clockwise';
    const raw = getTaxiwayDirection(rw);
    const d = normalizeRwDirectionValue(raw);
    if (d === 'clockwise' || d === 'counter_clockwise') return d;
    return 'clockwise';
  }
  /**
   * F2: same semantics as checkboxes, but an explicit `allowedRwDirections: []` (both unchecked) means
   * "no use" — do not re-expand to the default both-way list (pathfinding may still do that for legacy).
   */
  function isRunwayExitDirAllowedForArrivalFilter2(exitTw, runwayDir) {
    const d = normalizeRwDirectionValue(runwayDir);
    if (d !== 'clockwise' && d !== 'counter_clockwise') return true;
    if (!exitTw || exitTw.pathType !== 'runway_exit') return false;
    if (Object.prototype.hasOwnProperty.call(exitTw, 'allowedRwDirections')) {
      const arr = normalizeAllowedRunwayDirections(exitTw.allowedRwDirections);
      if (arr.length === 0) return false;
      return arr.indexOf(d) >= 0;
    }
    return isRunwayExitDirectionAllowed(exitTw, d);
  }
  function getRunwayExitAllowedDirectionsFromPanel() {
    const out = [];
    const container = document.getElementById('runwayExitAllowedDirection');
    if (!container) return out;
    container.querySelectorAll('.runway-exit-dir-check').forEach(function(ch) {
      if (!ch.checked) return;
      const value = String(ch.getAttribute('data-item-id') || '').trim();
      if (value === 'clockwise' || value === 'counter_clockwise') out.push(value);
    });
    return out;
  }

  const _rwy = _tiers.runway || {};
  const _sepUi = (_rwy.separationUi && typeof _rwy.separationUi === 'object') ? _rwy.separationUi : {};
  const RSEP_ARRDEP_BOOST_SEC = Math.max(0, Number(_sepUi.arrDepDefaultBoostSec) || 50);
  const RSEP_COLOR_THRESHOLDS = (function() {
    const arr = _sepUi.inputColorThresholdsSec;
    if (Array.isArray(arr) && arr.length) {
      return arr.map(x => Number(x)).filter(x => isFinite(x) && x > 0).sort((a, b) => a - b);
    }
    return [90, 120, 150];
  })();
  const RSEP_LEGEND_LAB = (_sepUi.legendLabels && typeof _sepUi.legendLabels === 'object') ? _sepUi.legendLabels : {};
  function rsepLegendFmt(tpl, a0, a1) {
    let s = String(tpl || '');
    if (a1 != null && s.indexOf('{1}') >= 0) return s.replace('{0}', String(a0)).replace('{1}', String(a1));
    return s.replace('{0}', String(a0));
  }
  const RSEP_COLOR_STYLES = [
    { bg: '#0d2018', color: '#68d391', border: '#68d39155' },
    { bg: '#0d1a28', color: '#63b3ed', border: '#63b3ed55' },
    { bg: '#1e1e08', color: '#f6e05e', border: '#f6e05e55' },
    { bg: '#280d0d', color: '#fc8181', border: '#fc818155' },
  ];
  const _stds = _rwy.standards || {};
  const RSEP_STD_CATS = {
    'ICAO': (_stds.ICAO && _stds.ICAO.categories) ? _stds.ICAO.categories : ['J','H','M','L'],
    'RECAT-EU': (_stds['RECAT-EU'] && _stds['RECAT-EU'].categories) ? _stds['RECAT-EU'].categories : ['A','B','C','D','E','F'],
  };
  const RSEP_SEQ_TYPES = Object.assign({ 'ARR→ARR': 'matrix', 'DEP→DEP': 'matrix', 'ARR→DEP': 'lead-1d', 'DEP→ARR': 'trail-1d' }, _sepUi.seqTypes || {});
  const RSEP_MODE_SEQS = (function() {
    const def = { ARR: ['ARR→ARR'], DEP: ['DEP→DEP'], MIX: ['ARR→ARR','DEP→DEP','ARR→DEP','DEP→ARR'] };
    const ms = _sepUi.modeSequences || {};
    const out = {};
    ['ARR','DEP','MIX'].forEach(k => {
      const a = ms[k];
      out[k] = (Array.isArray(a) && a.length) ? a.slice() : def[k].slice();
    });
    return out;
  })();
  const RSEP_DEFAULTS = {};
  ['ICAO','RECAT-EU'].forEach(k => {
    const s = _stds[k];
    if (!s) return;
    RSEP_DEFAULTS[k] = { ...(s.separationDefaults || {}), ROT: s.ROT || {} };
  });
  if (!RSEP_DEFAULTS['ICAO'] || !Object.keys(RSEP_DEFAULTS['ICAO']).length) {
    RSEP_DEFAULTS['ICAO'] = { 'ARR→ARR': { J:{J:90,H:120,M:180,L:240}, H:{J:90,H:90,M:120,L:180}, M:{J:90,H:90,M:90,L:180}, L:{J:90,H:90,M:90,L:90} }, 'DEP→DEP': { J:{J:90,H:120,M:180,L:180}, H:{J:90,H:90,M:120,L:120}, M:{J:90,H:90,M:90,L:90}, L:{J:90,H:90,M:90,L:90} }, 'ARR→DEP': {J:90,H:80,M:65,L:50}, 'DEP→ARR': {J:60,H:60,M:70,L:90}, ROT: {J:70,H:65,M:55,L:40} };
  }
  if (!RSEP_DEFAULTS['RECAT-EU'] || !Object.keys(RSEP_DEFAULTS['RECAT-EU']).length) {
    RSEP_DEFAULTS['RECAT-EU'] = { 'ARR→ARR': { A:{A:80,B:100,C:120,D:140,E:160,F:180}, B:{A:80,B:80,C:100,D:120,E:120,F:140}, C:{A:80,B:80,C:80,D:100,E:100,F:120}, D:{A:80,B:80,C:80,D:80,E:80,F:100}, E:{A:80,B:80,C:80,D:80,E:80,F:100}, F:{A:80,B:80,C:80,D:80,E:80,F:80} }, 'DEP→DEP': { A:{A:80,B:100,C:120,D:120,E:120,F:140}, B:{A:80,B:80,C:100,D:100,E:100,F:120}, C:{A:80,B:80,C:80,D:80,E:80,F:100}, D:{A:80,B:80,C:80,D:80,E:80,F:80}, E:{A:80,B:80,C:80,D:80,E:80,F:80}, F:{A:80,B:80,C:80,D:80,E:80,F:80} }, 'ARR→DEP': {A:80,B:70,C:60,D:55,E:50,F:45}, 'DEP→ARR': {A:55,B:55,C:60,D:65,E:70,F:80}, ROT: {A:65,B:60,C:55,D:50,E:45,F:40} };
  }
  const RSEP_STANDARDS = { 'ICAO': { ROT: RSEP_DEFAULTS['ICAO'] && RSEP_DEFAULTS['ICAO'].ROT ? RSEP_DEFAULTS['ICAO'].ROT : {} }, 'RECAT-EU': { ROT: RSEP_DEFAULTS['RECAT-EU'] && RSEP_DEFAULTS['RECAT-EU'].ROT ? RSEP_DEFAULTS['RECAT-EU'].ROT : {} } };
  const RSEP_CAT_LABELS = {
    'ICAO': (_stds.ICAO && _stds.ICAO.categoryLabels) ? _stds.ICAO.categoryLabels : { J:'Super', H:'Heavy', M:'Medium', L:'Light' },
    'RECAT-EU': (_stds['RECAT-EU'] && _stds['RECAT-EU'].categoryLabels) ? _stds['RECAT-EU'].categoryLabels : { A:'Super-Heavy', B:'Upper-Heavy', C:'Lower-Heavy', D:'Medium', E:'Light', F:'Very-Light' },
  };
  const RSEP_SEQ_META = _rwy.seqMeta || {
    'ARR→ARR': { driver: 'Wake of leading arrival aircraft', refPoint: 'Touchdown / final approach point of the leading arrival', input: 'Lead (arrival) × Trail (arrival) matrix input' },
    'DEP→DEP': { driver: 'Wake of leading departure aircraft', refPoint: 'Take-off / runway entry point of the leading departure', input: 'Lead (departure) × Trail (departure) matrix input' },
    'ARR→DEP': { driver: 'Leading aircraft ROT (runway occupancy time)', refPoint: 'Trailing aircraft: time from lineup to gear-off (lineup–gear-off)', input: 'Lead arrival category — 1D separation inputs' },
    'DEP→ARR': { driver: 'Wake / ROT of leading departure', refPoint: 'Runway vacation / ROT end of the leading departure', input: 'Trail (arrival category) 1‑D input' },
  };
  function rsepGetCatLabel(stdKey, cat) {
    const t = RSEP_CAT_LABELS[stdKey];
    if (!t) return '';
    return t[cat] || '';
  }
  function rsepGetSeqMeta(seq) {
    return RSEP_SEQ_META[seq] || null;
  }
  function _rsepStringValue(value) {
    return value != null ? String(value) : '';
  }
  function _rsepMakeCategoryValues(cats, src, asMatrix) {
    const out = {};
    cats.forEach(leadCat => {
      if (!asMatrix) {
        out[leadCat] = _rsepStringValue(src && src[leadCat]);
        return;
      }
      out[leadCat] = {};
      cats.forEach(trailCat => {
        out[leadCat][trailCat] = _rsepStringValue(src && src[leadCat] && src[leadCat][trailCat]);
      });
    });
    return out;
  }
  function rsepMakeMatrix(cats, src) {
    return _rsepMakeCategoryValues(cats, src, true);
  }
  function rsepMake1D(cats, src) {
    return _rsepMakeCategoryValues(cats, src, false);
  }
  function rsepMakeSeqData(stdKey) {
    const cats = RSEP_STD_CATS[stdKey] || [];
    const def = RSEP_DEFAULTS[stdKey] || {};
    const arrDep = rsepMake1D(cats, def['ARR→DEP']);
    const boost = RSEP_ARRDEP_BOOST_SEC;
    cats.forEach(function(c) {
      const s = arrDep[c];
      if (s === '' || s == null) return;
      const n = Number(s);
      if (isFinite(n)) arrDep[c] = String(Math.round(n + boost));
    });
    return {
      'ARR→ARR': rsepMakeMatrix(cats, def['ARR→ARR']),
      'DEP→DEP': rsepMakeMatrix(cats, def['DEP→DEP']),
      'ARR→DEP': arrDep,
      'DEP→ARR': rsepMake1D(cats, def['DEP→ARR']),
    };
  }

  function rsepColorForValue(val) {
    const n = Number(val);
    if (!isFinite(n) || val === '' || val == null) {
      return { bg: '#1a1a1a', color: '#e5e7eb', border: '#444444' };
    }
    const th = RSEP_COLOR_THRESHOLDS;
    for (let i = 0; i < th.length; i++) {
      if (n < th[i]) return RSEP_COLOR_STYLES[i] || RSEP_COLOR_STYLES[RSEP_COLOR_STYLES.length - 1];
    }
    return RSEP_COLOR_STYLES[th.length] || RSEP_COLOR_STYLES[RSEP_COLOR_STYLES.length - 1];
  }
  function rsepLegendHtml(filled, total) {
    const th = RSEP_COLOR_THRESHOLDS;
    const countColor = filled === total ? '#68d391' : '#9ca3af';
    let html = '<div style="display:flex;align-items:center;gap:12px;margin-top:4px;margin-bottom:4px;font-size:10px;color:#9ca3af;">';
    const lab = RSEP_LEGEND_LAB;
    if (th.length) {
      const st0 = rsepColorForValue(Math.max(0, th[0] - 1));
      html += '<span><span style="display:inline-block;width:10px;height:10px;background:' + st0.bg + ';border-radius:2px;margin-right:4px;"></span><span style="color:' + st0.color + ';">' + escapeHtml(rsepLegendFmt(lab.ltFirst || '<{0}s', th[0])) + '</span></span>';
      for (let i = 1; i < th.length; i++) {
        const lo = th[i - 1], hi = th[i];
        const mid = lo + (hi - lo) / 2;
        const st = rsepColorForValue(mid);
        const text = rsepLegendFmt(lab.rangeMid || '{0}–{1}s', lo, hi - 1);
        html += '<span><span style="display:inline-block;width:10px;height:10px;background:' + st.bg + ';border-radius:2px;margin-right:4px;"></span><span style="color:' + st.color + ';">' + escapeHtml(text) + '</span></span>';
      }
      const lastT = th[th.length - 1];
      const stL = rsepColorForValue(lastT + 1000);
      html += '<span><span style="display:inline-block;width:10px;height:10px;background:' + stL.bg + ';border-radius:2px;margin-right:4px;"></span><span style="color:' + stL.color + ';">' + escapeHtml(rsepLegendFmt(lab.gteLast || '≥{0}s', lastT)) + '</span></span>';
    }
    html += '<span style="margin-left:4px;color:' + countColor + ';">' + filled + '/' + total + '</span>';
    html += '</div>';
    return html;
  }
  function rsepMakeConfig(stdKey) {
    const std = RSEP_STANDARDS[stdKey] || RSEP_STANDARDS['ICAO'];
    const cats = RSEP_STD_CATS[stdKey];
    const rot = std.ROT || {};
    const rotCopy = {};
    const boost = RSEP_ARRDEP_BOOST_SEC;
    cats.forEach(function(c) {
      if (rot[c] == null || rot[c] === '') rotCopy[c] = '';
      else {
        const n = Number(rot[c]);


        rotCopy[c] = isFinite(n) ? String(Math.round(n + boost)) : String(rot[c]);
      }
    });
    return {
      standard: stdKey,
      mode: 'MIX',
      activeSeq: 'ARR→ARR',
      seqData: rsepMakeSeqData(stdKey),
      rot: rotCopy,
    };
  }
  function rsepGetConfigForRunway(rw) {
    if (!rw) return null;
    if (!rw.rwySepConfig) {
      rw.rwySepConfig = rsepMakeConfig('ICAO');
    }
    const cfg = rw.rwySepConfig;
    if (!RSEP_STD_CATS[cfg.standard]) {
      rw.rwySepConfig = rsepMakeConfig('ICAO');
      return rw.rwySepConfig;
    }
    return cfg;
  }
  let dpr = window.devicePixelRatio || 1;
  let ctx = (canvas && typeof canvas.getContext === 'function') ? canvas.getContext('2d') : null;
  let layoutDrawCanvas = canvas;
  function layoutUseForegroundOverlay() {
    return !!(overlayCtx && overlayCanvas && layoutDrawCanvas === canvas);
  }

  function screenToWorld(sx, sy) {
    return [(sx - state.panX) / state.scale, (sy - state.panY) / state.scale];
  }
  function worldToScreenCanvas(wx, wy) {
    return [wx * state.scale + state.panX, wy * state.scale + state.panY];
  }
  function cellToPixel(col, row) { return [col * CELL_SIZE, row * CELL_SIZE]; }
  function getTaxiwayAvgMoveVelocityForPath(path) {
    if (path && typeof path.avgMoveVelocity === 'number' && isFinite(path.avgMoveVelocity) && path.avgMoveVelocity > 0)
      return Math.max(1, Math.min(50, path.avgMoveVelocity));
    const el = document.getElementById('taxiwayAvgMoveVelocity');
    const v = el ? Number(el.value) : 10;
    return (typeof v === 'number' && isFinite(v) && v > 0) ? Math.max(1, Math.min(50, v)) : 10;
  }
  function roundToStep(value, step) {
    const n = Number(value);
    const s = Number(step);
    if (!isFinite(n)) return 0;
    if (!isFinite(s) || s <= 0) return n;
    return Math.round(n / s) * s;
  }
  function clampToGridBounds(col, row) {
    const c = Math.max(0, Math.min(GRID_COLS, Number(col) || 0));
    const r = Math.max(0, Math.min(GRID_ROWS, Number(row) || 0));
    return [c, r];
  }
  function pixelToCell(x, y) {
    const cs = (typeof CELL_SIZE === 'number' && CELL_SIZE > 0) ? CELL_SIZE : 20;
    const snappedCol = roundToStep(x / cs, GRID_SNAP_STEP_CELL);
    const snappedRow = roundToStep(y / cs, GRID_SNAP_STEP_CELL);
    return clampToGridBounds(snappedCol, snappedRow);
  }
  function worldPointToCellPoint(wx, wy, snapToGrid) {
    const cs = (typeof CELL_SIZE === 'number' && CELL_SIZE > 0) ? CELL_SIZE : 20;
    const step = snapToGrid ? GRID_SNAP_STEP_CELL : FREE_DRAW_STEP_CELL;
    const col = roundToStep(wx / cs, step);
    const row = roundToStep(wy / cs, step);
    const clamped = clampToGridBounds(col, row);
    return { col: clamped[0], row: clamped[1] };
  }
  function worldPointToPixel(wx, wy, snapToGrid) {
    const pt = worldPointToCellPoint(wx, wy, snapToGrid);
    return cellToPixel(pt.col, pt.row);
  }
  const ICAO_STAND_SIZE_M = (function() {
    const m = _layoutTier.standSizesMByIcaoCategory;
    if (m && typeof m === 'object') {
      const o = {};
      Object.keys(m).forEach(k => { o[k] = Number(m[k]); });
      return o;
    }
    return { A: 20, B: 30, C: 40, D: 50, E: 60, F: 80 };
  })();
  function getStandSizeMeters(cat) { return ICAO_STAND_SIZE_M[cat] || 40; }
  const STAND_CONFIG_ROW_BY_CODE = (function() {
    const raw = _layoutTier.standConfig;
    const out = {};
