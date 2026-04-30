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
  function syncProSimButtonFromDesignerPageState() {
    const btn = document.getElementById('btnGlobalUpdate');
    const dot = document.getElementById('globalUpdateSyncDot');
    const ban = document.getElementById('arrRetFailedBanner');
    const banT = document.getElementById('arrRetFailedBannerText');
    const failedRegs = getArrRetFailedRegsForProSimUi();
    const hasRetFail = failedRegs.length > 0;
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
    const allow = !!state.designerPageUpdateFresh && !hasRetFail;
    if (btn) {
      btn.disabled = !allow;
      btn.classList.toggle('global-update-blocked-arr-ret', hasRetFail);
      if (!state.designerPageUpdateFresh) {
        btn.setAttribute('title', 'Run Update first (green sync) to refresh the path graph and views, then use Pro Sim.');
      } else if (hasRetFail) {
        const n = failedRegs.length;
        const shortList = n > 5 ? (failedRegs.slice(0, 3).join(', ') + ', etc. (' + n + ' total)') : failedRegs.join(', ');
        btn.setAttribute('title', 'Pro Sim is disabled: no valid runway exit. ' + shortList);
      } else {
        btn.setAttribute('title', 'Run airside_sim on the server; saves layoutName_sim_result.json under Result_storage');
      }
    }
    if (dot) {
      if (hasRetFail) {
        dot.classList.remove('fresh');
        dot.classList.add('stale');
        dot.setAttribute('title', 'Runway exit failure — resolve all arrival RET issues before Pro Sim.');
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
    const proSimUiFresh = !hasRetFail && !!state.globalUpdateFresh;
    const allowPlay = !!state.hasSimulationResult && proSimUiFresh;
    if (playDock) {
      playDock.disabled = !allowPlay;
    }
    let playbackMemSync = false;
    if (!allowPlay) {
      if (typeof evictFlightPlaybackTimelinesWhenPlayBlocked === 'function') {
        playbackMemSync = evictFlightPlaybackTimelinesWhenPlayBlocked();
      }
      state.simPlaybackDockVisible = false;
      state.simPlaying = false;
      state.simSliderScrubbing = false;
      if (typeof ensureSimLoop === 'function') ensureSimLoop._playKick = false;
      if (typeof applySimPlaybackBarDomVisibility === 'function') applySimPlaybackBarDomVisibility();
    } else {
      if (typeof rehydrateFlightPlaybackTimelinesAfterPlayAllowed === 'function') {
        playbackMemSync = rehydrateFlightPlaybackTimelinesAfterPlayAllowed();
      }
    }
    if (playbackMemSync) {
      if (typeof draw === 'function') draw();
      if (typeof update3DSceneWhenVisible === 'function') update3DSceneWhenVisible();
    }
    if (typeof syncMapTypePopoverFromState === 'function') syncMapTypePopoverFromState();
  }
  function redrawLayoutAfterEdit() {
    if (typeof bumpScheduleRetExitDistCache === 'function') bumpScheduleRetExitDistCache();
    // Full reset rebuilds junctions in draw(); manual-sync mode keeps last graph for display until Update / Pro Sim.
    if (PATH_GRAPH_SYNC_ONLY_ON_EXPLICIT_ACTION) {
      invalidatePathGraphCache(false);
    } else {
      invalidatePathGraphCache(true);
    }
    if (typeof markGlobalUpdateStale === 'function') markGlobalUpdateStale();
    if (typeof markDesignerPageUpdateStale === 'function') markDesignerPageUpdateStale();
    if (typeof draw === 'function') draw();
    if (typeof update3DSceneWhenVisible === 'function') update3DSceneWhenVisible();
  }
  function setGlobalUpdateProgressUi(visible, label, pct) {
    const ov = document.getElementById('globalUpdateOverlay');
    const fill = document.getElementById('globalUpdateProgressFill');
    const lab = document.getElementById('globalUpdateOverlayLabel');
    const btn = document.getElementById('btnGlobalUpdate');
    if (!ov) return;
    if (visible) {
      ov.classList.add('is-visible');
      ov.setAttribute('aria-hidden', 'false');
      if (lab && label != null) lab.textContent = label;
      if (fill && pct != null) fill.style.width = Math.max(0, Math.min(100, pct)) + '%';
      if (btn) btn.disabled = true;
    } else {
      ov.classList.remove('is-visible');
      ov.setAttribute('aria-hidden', 'true');
      if (fill) fill.style.width = '0%';
      if (typeof syncProSimButtonFromDesignerPageState === 'function') syncProSimButtonFromDesignerPageState();
      else if (btn) btn.disabled = false;
    }
  }
  function scheduleAfterPaint(fn) {
    requestAnimationFrame(function() {
      requestAnimationFrame(function() { setTimeout(fn, 0); });
    });
  }
  const DEFAULT_AIRLINE_CODES = (function() {
    const a = _flightTier.defaultAirlineCodes;
    return (Array.isArray(a) && a.length) ? a.map(String) : ['KE', '7C', 'DL'];
  })();
  const PATH_LAYOUT_MODES = ['runwayPath', 'runwayTaxiway', 'taxiway'];
  function pathTypeFromLayoutMode(layoutMode) {
    if (layoutMode === 'runwayPath') return 'runway';
    if (layoutMode === 'runwayTaxiway') return 'runway_exit';
    if (layoutMode === 'taxiway') return 'taxiway';
    return 'taxiway';
  }


  function layoutModeFromPathType(pt) {
    if (pt === 'runway') return 'runwayPath';
    if (pt === 'runway_exit') return 'runwayTaxiway';
    if (pt === 'apron_taxiway') return 'taxiway';
    if (pt === 'general_queue_taxiway') return 'taxiway';
    return 'taxiway';
  }
  function isPathLayoutMode(m) {
    return PATH_LAYOUT_MODES.indexOf(m) >= 0;
  }
  function standHasApronTaxiwayLink(standId) {
    if (standId == null || standId === '') return false;
    const links = state.apronLinks || [];
    const tws = state.taxiways || [];
    for (let i = 0; i < links.length; i++) {
      const lk = links[i];
      if (!lk || lk.pbbId !== standId) continue;
      const tid = lk.taxiwayId;
      for (let j = 0; j < tws.length; j++) {
        if (tws[j] && tws[j].id === tid) return true;
      }
    }
    return false;
  }
  function ganttESeriesMinutesFromTimelineMeta(f) {
    const m = f && f.timeline_meta;
    if (!m || typeof m !== 'object') {
      return { eldt: NaN, eibt: NaN, eobt: NaN, etot: NaN };
    }
    const toMin = function(sec) {
      const n = sec != null ? Number(sec) : NaN;
      return (isFinite(n) ? n / 60 : NaN);
    };
    return {
      eldt: toMin(m.eldtSec),
      eibt: toMin(m.eibtSec),
      eobt: toMin(m.eobtSec),
      etot: toMin(m.etotSec),
    };
  }
  function settingModeValueForHit(hit) {
    if (!hit || !hit.type) return null;
    if (hit.type === 'terminal') return 'terminal';
    if (hit.type === 'pbb') return 'pbb';
    if (hit.type === 'remote') return 'remote';
    if (hit.type === 'tempStand') return 'tempStand';
    if (hit.type === 'holdingPoint') return 'holdingPoint';
    if (hit.type === 'taxiway') return layoutModeFromPathType((hit.obj && hit.obj.pathType) || 'taxiway');
    if (hit.type === 'apronLink') return 'apronTaxiway';
    if (hit.type === 'layoutMarker') return 'marker';
    return null;
  }
  function cancelActiveLayoutDrawingState() {
    state.pbbDrawing = false;
    state.remoteDrawing = false;
    state.tempStandDrawing = false;
    state.holdingPointDrawing = false;
    state.previewHoldingPoint = null;
    state.apronLinkDrawing = false;
    state.apronLinkTemp = null;
    state.apronLinkMidpoints = [];
    state.apronLinkPointerWorld = null;
