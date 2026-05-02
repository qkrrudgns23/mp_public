    function filt(arr) {
      if (!Array.isArray(arr)) return arr;
      return arr.filter(function(pt) { return !pointNearDeletedTw(pt); });
    }
    if (Array.isArray(g.validJunctions)) g.validJunctions = filt(g.validJunctions);
    if (Array.isArray(g.connectedJunctions)) g.connectedJunctions = filt(g.connectedJunctions);
    if (Array.isArray(g.junctions)) g.junctions = filt(g.junctions);
    if (Array.isArray(g.disconnectedValidJunctions)) g.disconnectedValidJunctions = filt(g.disconnectedValidJunctions);
  }
  function markPathGraphJunctionStaleShellAfterLayoutEdit() {
    cancelPathGraphRebuildTimer();
    state.pathGraphCache = {
      __junctionStale: true,
      validJunctions: [],
      connectedJunctions: [],
      disconnectedValidJunctions: [],
      junctions: [],
      nodes: [],
      edges: [],
      adj: [],
      edgeMap: {},
      runwayNodeIndicesById: {},
      standIdToNodeIndex: {}
    };
    state.pathGraphCacheValid = true;
    state.pathGraphCacheSig = computeTaxiwaysGraphSig();
    state.pathGraphCacheDirty = true;
    state.pathGraphInvalidatedAtMs = Date.now();
  }
  function graphSigParseRecords(sig) {
    const m = {};
    if (!sig || typeof sig !== 'string') return m;
    const chunks = sig.split('||');
    for (let i = 0; i < chunks.length; i++) {
      const rec = chunks[i];
      if (!rec) continue;
      const pipe = rec.indexOf('|');
      const id = pipe >= 0 ? rec.slice(0, pipe) : rec;
      if (id) m[id] = rec;
    }
    return m;
  }
  function graphSigTaxiwayDiff(oldSig, newSig) {
    const o = graphSigParseRecords(oldSig);
    const n = graphSigParseRecords(newSig);
    const removed = [];
    const changed = [];
    Object.keys(o).forEach(function(id) {
      if (!(id in n)) removed.push(id);
    });
    Object.keys(n).forEach(function(id) {
      if (!(id in o)) changed.push(id);
      else if (o[id] !== n[id]) changed.push(id);
    });
    return { removed: removed, changed: changed };
  }
  function cloneFlightsWithoutPathPolylineCache(flights) {
    return (flights || []).map(function(f) {
      const raw = JSON.parse(JSON.stringify(f));
      delete raw.cachedArrPathPts;
      delete raw.cachedDepPathPts;
      delete raw._pathPolylineCacheRev;
      delete raw._pathPolylineArrRetKey;
      return raw;
    });
  }
  function markGlobalUpdateStale() {
    state.globalUpdateFresh = false;
    state.simPlaying = false;
    state.simSliderScrubbing = false;
    if (typeof ensureSimLoop === 'function') ensureSimLoop._playKick = false;
    bumpPathPolylineCacheRev();
    state.rwySepPanelDirty = true;
    bumpRwySepSnapshotStaleGen();
    if (typeof clearAllFlightTimelines === 'function') clearAllFlightTimelines({ keepDesResultTimelines: true });
    if (typeof applySimPlaybackBarDomVisibility === 'function') applySimPlaybackBarDomVisibility();
    if (typeof syncProSimButtonFromDesignerPageState === 'function') syncProSimButtonFromDesignerPageState();
  }
  function markGlobalUpdateFresh() {
    state.globalUpdateFresh = true;
    if (typeof applySimPlaybackBarDomVisibility === 'function') applySimPlaybackBarDomVisibility();
    if (typeof syncProSimButtonFromDesignerPageState === 'function') syncProSimButtonFromDesignerPageState();
  }
  function markProSimSyncStaleFromSchedule() {
    state.globalUpdateFresh = false;
    state.simPlaying = false;
    state.simSliderScrubbing = false;
    if (typeof ensureSimLoop === 'function') ensureSimLoop._playKick = false;
    if (typeof applySimPlaybackBarDomVisibility === 'function') applySimPlaybackBarDomVisibility();
    if (typeof syncProSimButtonFromDesignerPageState === 'function') syncProSimButtonFromDesignerPageState();
  }
  function markDesignerPageUpdateStale() {
    state.designerPageUpdateFresh = false;
    const dot = document.getElementById('designerPageUpdateSyncDot');
    if (dot) {
      dot.classList.remove('fresh');
      dot.classList.add('stale');
      dot.setAttribute('title', '레이아웃/객체 변경됨 — Update를 눌러 경로 그래프·뷰를 동기화하세요');
    }
    if (typeof syncProSimButtonFromDesignerPageState === 'function') syncProSimButtonFromDesignerPageState();
  }
  function markDesignerPageUpdateFresh() {
    state.designerPageUpdateFresh = true;
    const dot = document.getElementById('designerPageUpdateSyncDot');
    if (dot) {
      dot.classList.remove('stale');
      dot.classList.add('fresh');
      dot.setAttribute('title', 'Update 기준으로 경로·뷰가 최신입니다');
    }
    if (typeof syncProSimButtonFromDesignerPageState === 'function') syncProSimButtonFromDesignerPageState();
  }
  /** English multi-line (≤5 aircraft) or summary (6+) for arrival RET failure dock banner. */
  function formatArrRetFailedBannerEnglish(regs) {
    const n = (regs && regs.length) || 0;
    if (n < 1) return '';
    if (n <= 5) {
      return regs.map(function(reg) {
        return String(reg) + ': Runway exit assignment failed.';
      }).join('\n');
    }
    const head = regs.slice(0, 3).join(', ');
    return head + ', etc. — ' + n + ' aircraft failed.';
  }
  /**
   * Pro Sim: allowed only when Update is fresh (green) and no arrival Runway exit (RET) failures
   * (`arrRetFailed` on non-departure legs). RET failure banner is in `#gridLeftFloatingStack`, separate from `#object-info-dock`.
   */
  function getArrRetFailedRegsForProSimUi() {
    const failedRegs = [];
    (state.flights || []).forEach(function(f) {
      if (!f || f.arrDep === 'Dep') return;
      if (f.arrRetFailed) {
        const r = String(f.reg != null && String(f.reg).trim() !== '' ? f.reg : (f.flightNumber || f.id || '')).trim() || '—';
        if (failedRegs.indexOf(r) < 0) failedRegs.push(r);
      }
    });
    return failedRegs;
  }
  function flightApronIntervalsForProSimBlock(f) {
    if (!f || flightBlockedLikeNoWay(f)) return [];
    const segs = typeof normalizeFlightApronStaySegments === 'function' ? normalizeFlightApronStaySegments(f) : [];
    const out = [];
    if (segs.length) {
      const count = segs.length;
      for (let i = 0; i < segs.length; i++) {
        const seg = segs[i];
        const standId = seg && seg.standId != null ? String(seg.standId) : '';
        const t0 = Number(seg && seg.sibtMin);
        const t1 = Number(seg && seg.sobtMin);
        if (!standId || !isFinite(t0) || !isFinite(t1) || t1 <= t0) continue;
        out.push({ f: f, standId: standId, t0: t0, t1: t1, segmentIdx: i, segmentCount: count });
      }
      return out;
    }
    const standId = f.standId != null ? String(f.standId) : '';
    const t0 = f.sibtMin != null ? Number(f.sibtMin) : Number(f.timeMin || 0);
    const t1 = f.sobtMin != null ? Number(f.sobtMin) : (t0 + Number(f.dwellMin || 0));
    return standId && isFinite(t0) && isFinite(t1) && t1 > t0
      ? [{ f: f, standId: standId, t0: t0, t1: t1, segmentIdx: 0, segmentCount: 1 }]
      : [];
  }
  function getApronDuplicatedRegsForProSimUi() {
    const issues = [];
    const seen = new Set();
    const intervalsByStand = {};
    function addIssue(f, reason) {
      if (!f) return;
      const reg = String(f.reg != null && String(f.reg).trim() !== '' ? f.reg : (f.flightNumber || f.id || '')).trim() || '—';
      const key = reg + '|' + reason;
      if (seen.has(key)) return;
      seen.add(key);
      issues.push({ reg: reg, reason: reason });
    }
    (state.flights || []).forEach(function(f) {
      const intervals = flightApronIntervalsForProSimBlock(f);
      intervals.forEach(function(it) {
        const stand = typeof findStandById === 'function' ? findStandById(it.standId) : null;
        if (stand && typeof flightCanUseStandForSegment === 'function' && !flightCanUseStandForSegment(f, stand, it.segmentIdx, it.segmentCount)) {
          addIssue(f, 'Invalid apron/building assignment.');
        }
        if (!intervalsByStand[it.standId]) intervalsByStand[it.standId] = [];
        intervalsByStand[it.standId].push(it);
      });
    });
    Object.keys(intervalsByStand).forEach(function(standId) {
      const arr = intervalsByStand[standId];
      for (let i = 0; i < arr.length; i++) {
        for (let j = i + 1; j < arr.length; j++) {
          const a = arr[i], b = arr[j];
          if (a.f && b.f && a.f.id === b.f.id) continue;
          if (a.t0 < b.t1 && b.t0 < a.t1) {
            addIssue(a.f, 'Duplicated apron time window.');
            addIssue(b.f, 'Duplicated apron time window.');
          }
        }
      }
    });
    return issues;
  }
  function formatApronDuplicatedBannerEnglish(issues) {
    const n = (issues && issues.length) || 0;
    if (n < 1) return '';
    if (n <= 5) {
      return issues.map(function(it) {
        return String(it.reg) + ': ' + String(it.reason || 'Apron duplicated.');
      }).join('\n');
    }
    const head = issues.slice(0, 3).map(function(it) { return it.reg; }).join(', ');
    return head + ', etc. — ' + n + ' apron assignment issue(s).';
  }
  function syncProSimButtonFromDesignerPageState() {
    const btn = document.getElementById('btnGlobalUpdate');
    const dot = document.getElementById('globalUpdateSyncDot');
    const playDot = document.getElementById('playbackFreshSyncDot');
    const ban = document.getElementById('arrRetFailedBanner');
    const banT = document.getElementById('arrRetFailedBannerText');
    const apronBan = document.getElementById('apronDuplicatedBanner');
    const apronBanT = document.getElementById('apronDuplicatedBannerText');
    const failedRegs = getArrRetFailedRegsForProSimUi();
    const hasRetFail = failedRegs.length > 0;
    const apronIssues = getApronDuplicatedRegsForProSimUi();
    const hasApronDuplicated = apronIssues.length > 0;
    if (ban && banT) {
      if (hasRetFail) {
        ban.hidden = false;
        ban.setAttribute('aria-hidden', 'false');
        banT.textContent = formatArrRetFailedBannerEnglish(failedRegs);
      } else {
        ban.hidden = true;
        ban.setAttribute('aria-hidden', 'true');
        banT.textContent = '';
      }
    }
    if (apronBan && apronBanT) {
      if (hasApronDuplicated) {
        apronBan.hidden = false;
        apronBan.setAttribute('aria-hidden', 'false');
        apronBanT.textContent = formatApronDuplicatedBannerEnglish(apronIssues);
      } else {
        apronBan.hidden = true;
        apronBan.setAttribute('aria-hidden', 'true');
        apronBanT.textContent = '';
      }
    }
    const dlBan = document.getElementById('deadlockGhostBanner');
    const dlBanT = document.getElementById('deadlockGhostBannerText');
    const dlp = state.simDeadlockGhostPlayback || { events: [], bodyLines: '', resolveCount: 0 };
    const showDeadlock = !!state.hasSimulationResult && ((dlp.events && dlp.events.length > 0) || (dlp.resolveCount > 0));
    if (dlBan && dlBanT) {
      if (showDeadlock) {
        dlBan.hidden = false;
        dlBan.setAttribute('aria-hidden', 'false');
        dlBanT.textContent = dlp.bodyLines || (dlp.resolveCount > 0 ? ('Deadlock auto-resolve recorded ' + dlp.resolveCount + ' time(s).') : '');
      } else {
        dlBan.hidden = true;
        dlBan.setAttribute('aria-hidden', 'true');
        dlBanT.textContent = '';
      }
    }
    const allow = !!state.designerPageUpdateFresh && !hasRetFail && !hasApronDuplicated;
    if (btn) {
      btn.disabled = !allow;
      btn.classList.toggle('global-update-blocked-arr-ret', hasRetFail);
      btn.classList.toggle('global-update-blocked-apron', hasApronDuplicated);
      if (!state.designerPageUpdateFresh) {
        btn.setAttribute('title', 'Run Update first (green sync) to refresh the path graph and views, then use Pro Sim.');
      } else if (hasRetFail) {
        const n = failedRegs.length;
        const shortList = n > 5 ? (failedRegs.slice(0, 3).join(', ') + ', etc. (' + n + ' total)') : failedRegs.join(', ');
        btn.setAttribute('title', 'Pro Sim is disabled: no valid runway exit. ' + shortList);
      } else if (hasApronDuplicated) {
        const n = apronIssues.length;
        const shortList = n > 5 ? (apronIssues.slice(0, 3).map(function(it) { return it.reg; }).join(', ') + ', etc. (' + n + ' total)') : apronIssues.map(function(it) { return it.reg; }).join(', ');
        btn.setAttribute('title', 'Pro Sim is disabled: Apron duplicated. ' + shortList);
      } else {
        btn.setAttribute('title', 'Run airside_sim on the server; saves layoutName_sim_result.json under Result_storage');
      }
    }
    if (dot) {
      if (hasRetFail) {
        dot.classList.remove('fresh');
        dot.classList.add('stale');
        dot.setAttribute('title', 'Runway exit failure — resolve all arrival RET issues before Pro Sim.');
      } else if (hasApronDuplicated) {
        dot.classList.remove('fresh');
        dot.classList.add('stale');
        dot.setAttribute('title', 'Apron duplicated — resolve all red Apron Gantt bars before Pro Sim.');
      } else if (state.globalUpdateFresh) {
        dot.classList.remove('stale');
        dot.classList.add('fresh');
        dot.setAttribute('title', 'All views match the last Pro Sim run');
      } else {
        dot.classList.remove('fresh');
        dot.classList.add('stale');
        dot.setAttribute('title', 'Layout or schedule changed — run Pro Sim again to refresh (results apply when done)');
      }
    }
    const playDock = document.getElementById('btnShowPlayDock');
    const playbackFresh = !hasRetFail && !hasApronDuplicated && !!state.globalUpdateFresh;
    const allowPlay = !!state.hasSimulationResult && !hasRetFail && !hasApronDuplicated;
    if (playDock) {
      playDock.disabled = !allowPlay;
      if (!state.hasSimulationResult) {
        playDock.setAttribute('title', '시뮬레이션 결과가 있을 때 재생 바를 엽니다');
      } else if (hasRetFail) {
        playDock.setAttribute('title', 'Runway exit failure가 있어 재생을 막았습니다');
      } else if (hasApronDuplicated) {
        playDock.setAttribute('title', 'Apron duplicated가 있어 Pro Sim/재생을 막았습니다');
      } else if (playbackFresh) {
        playDock.setAttribute('title', '최신 Pro Sim 결과를 재생합니다');
      } else {
        playDock.setAttribute('title', '이전 Pro Sim 결과를 재생합니다 — 레이아웃 변경으로 최신 상태는 아닙니다');
      }
    }
    if (playDot) {
      if (playbackFresh) {
        playDot.classList.remove('stale');
        playDot.classList.add('fresh');
        playDot.setAttribute('title', 'Playback result matches the latest layout');
      } else {
