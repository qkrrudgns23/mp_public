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
  function playbackLastMotionUnitDirBeforeTime(track, tSec) {
    const eps = 0.08;
    const eps2 = eps * eps;
    if (!isCompactPlaybackTrack(track)) return null;
    const idx = compactPlaybackIndexAtTime(track, tSec, true);
    if (idx < 1) return null;
    for (let j = idx - 1; j >= 0; j--) {
      const p = compactPlaybackSampleAtIndex(track, j);
      const q = compactPlaybackSampleAtIndex(track, j + 1);
      if (!p || !q) continue;
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
      typeof snapSimTimeSecForSlider === 'function'
        ? snapSimTimeSecForSlider(Math.max(lo, Math.min(hi, tAbs)))
        : Math.max(lo, Math.min(hi, tAbs));
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
    function ghostSessionStartTimestampsSec(arr, gapSec) {
      const g = Math.max(30, Number(gapSec) || 120);
      const nums = [];
      for (let i = 0; i < arr.length; i++) {
        const v = Math.round(Number(arr[i]));
        if (isFinite(v)) nums.push(v);
      }
      if (!nums.length) return [];
      nums.sort(function(a, b) {
        return a - b;
      });
      const out = [nums[0]];
      for (let j = 1; j < nums.length; j++) {
        if (nums[j] - nums[j - 1] > g) out.push(nums[j]);
      }
      return out;
    }
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
      bodyLines = events
        .map(function(ev) {
          const timeStr = formatTotalSecondsToHHMMSS(ev.t_abs);
          const reg0 =
            ev.labels && ev.labels.length ? String(ev.labels[0]).trim() : '';
          const reg = reg0 || '—';
          return timeStr + '  ' + reg;
        })
        .join('\n');
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
      dot.setAttribute('title', 'Deadlock @ ' + formatTotalSecondsToHHMMSS(t) + ' — click to jump');
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
