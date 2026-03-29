          '<br>Code(ICAO): ' + (codeIcao || '—') + ' &nbsp; ICAO(J/H/M/L): ' + (icaoJhl || '—') + ' &nbsp; RECAT-EU: ' + (recatEu || '—') +
          '<br>Reg: ' + (o.reg || '—') +
          '<br>Airline Code: ' + (o.airlineCode || '—') + ' &nbsp; Flight Number: ' + (o.flightNumber || '—') +
          '<br>Dwell (Arr only): ' + (o.dwellMin || 0) + ' min';
      }
    } else
      objectInfoEl.textContent = 'Select an object on the grid or from the list.';
    renderObjectList();
  }

  function reset2DView() {
    let w = 0, h = 0;
    const rect = container.getBoundingClientRect();
    w = Number(rect.width) || 0;
    h = Number(rect.height) || 0;
    if (w <= 0 || h <= 0) {
      if (canvas) {
        w = canvas.clientWidth || canvas.width || 800;
        h = canvas.clientHeight || canvas.height || 600;
      } else {
        w = 800;
        h = 600;
      }
    }
    w = Math.max(1, w);
    h = Math.max(1, h);
    const maxX = GRID_COLS * CELL_SIZE;
    const maxY = GRID_ROWS * CELL_SIZE;
    const scaleX = w / maxX;
    const scaleY = h / maxY;
    const s = Math.min(scaleX, scaleY) * 0.9;
    state.scale = s;
    state.panX = (w - maxX * s) / 2;
    state.panY = (h - maxY * s) / 2;
    draw();
  }

  function resizeCanvas() {
    if (!container || !canvas || !ctx) return;
    const rect = container.getBoundingClientRect();
    const w = Math.max(1, Number(rect.width) || 0);
    const h = Math.max(1, Number(rect.height) || 0);
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    canvas.style.width = w + 'px';
    canvas.style.height = h + 'px';
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    invalidateGridUnderlay();
    safeDraw();
  }

  let _gridUnderlayCanvas = null;
  let _gridUnderlayDirty = true;
  function invalidateGridUnderlay() { _gridUnderlayDirty = true; }
  function rebuildGridUnderlay() {
    const maxX = GRID_COLS * CELL_SIZE, maxY = GRID_ROWS * CELL_SIZE;
    if (!_gridUnderlayCanvas) _gridUnderlayCanvas = document.createElement('canvas');
    _gridUnderlayCanvas.width = Math.max(1, Math.floor(maxX * dpr));
    _gridUnderlayCanvas.height = Math.max(1, Math.floor(maxY * dpr));
    const uctx = _gridUnderlayCanvas.getContext('2d');
    uctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    uctx.fillStyle = GRID_VIEW_BG;
    uctx.fillRect(0, 0, maxX, maxY);
    if (state.layoutImageOverlay && layoutImageBitmap) {
      const overlay = state.layoutImageOverlay;
      const [imgX, imgY] = cellToPixel(overlay.topLeftCol, overlay.topLeftRow);
      uctx.save();
      uctx.globalAlpha = state.showImage ? clampLayoutImageOpacity(overlay.opacity) : 0;
      uctx.imageSmoothingEnabled = true;
      uctx.drawImage(
        layoutImageBitmap,
        imgX,
        imgY,
        clampLayoutImageSize(overlay.widthM, GRID_LAYOUT_IMAGE_DEFAULTS.widthM),
        clampLayoutImageSize(overlay.heightM, GRID_LAYOUT_IMAGE_DEFAULTS.heightM)
      );
      uctx.restore();
    }
    _gridUnderlayDirty = false;
  }

  function drawGrid() {
    const w = canvas.width / dpr, h = canvas.height / dpr;
    ctx.save();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = GRID_VIEW_BG;
    ctx.fillRect(0, 0, w, h);
    ctx.restore();
    ctx.save();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.translate(state.panX, state.panY);
    ctx.scale(state.scale, state.scale);
    const maxX = GRID_COLS * CELL_SIZE, maxY = GRID_ROWS * CELL_SIZE;
    if (_gridUnderlayDirty) rebuildGridUnderlay();
    ctx.drawImage(_gridUnderlayCanvas, 0, 0, maxX, maxY);
    if (!state.showGrid) {
      ctx.restore();
      return;
    }
    const drawMinor = !(GRID_MINOR_GRID_MIN_SCALE > 0 && state.scale < GRID_MINOR_GRID_MIN_SCALE);
    const marginWorld = GRID_DRAW_VIEWPORT_MARGIN_CELLS * CELL_SIZE;
    const s = state.scale || 1;
    const minWx = (0 - state.panX) / s - marginWorld;
    const maxWx = (w - state.panX) / s + marginWorld;
    const minWy = (0 - state.panY) / s - marginWorld;
    const maxWy = (h - state.panY) / s + marginWorld;
    const cMin = Math.max(0, Math.floor(minWx / CELL_SIZE));
    const cMax = Math.min(GRID_COLS, Math.ceil(maxWx / CELL_SIZE));
    const rMin = Math.max(0, Math.floor(minWy / CELL_SIZE));
    const rMax = Math.min(GRID_ROWS, Math.ceil(maxWy / CELL_SIZE));
    for (let c = cMin; c <= cMax; c++) {
      const isMajor = (c % GRID_MAJOR_INTERVAL === 0);
      if (!isMajor && !drawMinor) continue;
      const x = c * CELL_SIZE;
      ctx.strokeStyle = isMajor
        ? ('rgba(' + GRID_MAJOR_LINE_RGB + ',' + GRID_MAJOR_LINE_OPACITY + ')')
        : ('rgba(' + GRID_MINOR_LINE_RGB + ',' + GRID_MINOR_LINE_OPACITY + ')');
      ctx.lineWidth = isMajor ? GRID_MAJOR_LINE_WIDTH : GRID_MINOR_LINE_WIDTH;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, maxY);
      ctx.stroke();
    }
    for (let r = rMin; r <= rMax; r++) {
      const isMajor = (r % GRID_MAJOR_INTERVAL === 0);
      if (!isMajor && !drawMinor) continue;
      const y = r * CELL_SIZE;
      ctx.strokeStyle = isMajor
        ? ('rgba(' + GRID_MAJOR_LINE_RGB + ',' + GRID_MAJOR_LINE_OPACITY + ')')
        : ('rgba(' + GRID_MINOR_LINE_RGB + ',' + GRID_MINOR_LINE_OPACITY + ')');
      ctx.lineWidth = isMajor ? GRID_MAJOR_LINE_WIDTH : GRID_MINOR_LINE_WIDTH;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(maxX, y);
      ctx.stroke();
    }
    ctx.fillStyle = '#aaa';
    ctx.font = '10px system-ui';
    ctx.fillText('0,0', 4, 2);
    const cx = (GRID_COLS * CELL_SIZE) / 2;
    const cy = (GRID_ROWS * CELL_SIZE) / 2;
    ctx.beginPath();
    ctx.fillStyle = '#ef4444';
    ctx.arc(cx, cy, CELL_SIZE * 0.15, 0, Math.PI * 2);
    ctx.fill();
    if (state.hoverCell != null) {
      const hc = state.hoverCell;
      const hx = hc.col * CELL_SIZE;
      const hy = hc.row * CELL_SIZE;
      ctx.beginPath();
      ctx.fillStyle = 'rgba(248, 113, 113, 0.45)';
      ctx.arc(hx, hy, CELL_SIZE * 0.2, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();
  }

  function drawPolygonHatch(points, strokeStyle, spacingPx) {
    if (!Array.isArray(points) || points.length < 3) return;
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    points.forEach(function(p) {
      minX = Math.min(minX, p[0]);
      maxX = Math.max(maxX, p[0]);
      minY = Math.min(minY, p[1]);
      maxY = Math.max(maxY, p[1]);
    });
    const span = Math.max(maxX - minX, maxY - minY);
    const pad = span + Math.max(40, spacingPx * 2);
    ctx.save();
    ctx.beginPath();
    ctx.moveTo(points[0][0], points[0][1]);
    for (let i = 1; i < points.length; i++) ctx.lineTo(points[i][0], points[i][1]);
    ctx.closePath();
    ctx.clip();
    ctx.strokeStyle = strokeStyle;
    ctx.lineWidth = 1.2;
    ctx.setLineDash([]);
    for (let offset = minX - pad; offset <= maxX + pad; offset += spacingPx) {
      ctx.beginPath();
      ctx.moveTo(offset, maxY + pad);
      ctx.lineTo(offset + (maxY - minY) + pad, minY - pad);
      ctx.stroke();
    }
    ctx.restore();
  }
  function drawTerminals() {
    ctx.save();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.translate(state.panX, state.panY);
    ctx.scale(state.scale, state.scale);
    state.terminals.forEach(term => {
      const isDrawingTerm = state.terminalDrawingId === term.id;
      if (term.vertices.length === 0 && !isDrawingTerm) return;
      const selected = state.selectedObject && state.selectedObject.type === 'terminal' && state.selectedObject.id === term.id;
      const buildingTheme = getBuildingTheme(term);
      const termPts = term.vertices.map(function(v) { return cellToPixel(v.col, v.row); });
      ctx.lineWidth = selected ? 3 : 2;
      ctx.strokeStyle = selected ? c2dObjectSelectedStroke() : buildingTheme.stroke;
      ctx.fillStyle = selected ? c2dObjectSelectedFill() : buildingTheme.fill;
      ctx.beginPath();
      for (let i = 0; i < termPts.length; i++) {
        const [x,y] = termPts[i];
        if (i === 0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
      }
      if (term.closed) {
        ctx.closePath();
        if (buildingTheme.fillEnabled) ctx.fill();
      }
      if (selected) {
        ctx.save();
        ctx.shadowColor = c2dObjectSelectedGlow();
        ctx.shadowBlur = c2dObjectSelectedGlowBlur();
        ctx.shadowOffsetX = 0;
        ctx.shadowOffsetY = 0;
      }
      ctx.stroke();
      if (selected) ctx.restore();
      if (term.closed && buildingTheme.hatch === 'diagonal' && buildingTheme.fillEnabled) {
        drawPolygonHatch(termPts, selected ? c2dObjectSelectedDashStroke() : buildingTheme.stroke, Math.max(10, CELL_SIZE * 0.6));
      }
      if (term.closed && term.vertices.length > 0) {
        let cx = 0, cy = 0;
        term.vertices.forEach(v => {
          const [px, py] = cellToPixel(v.col, v.row);
          cx += px; cy += py;
        });
        cx /= term.vertices.length;
        cy /= term.vertices.length;
        const label = term.name || term.id || 'Building';
        ctx.fillStyle = buildingTheme.labelFill;
        ctx.font = '12px system-ui';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(label, cx, cy);
      }
      term.vertices.forEach((v, i) => {
        const [x,y] = cellToPixel(v.col, v.row);
        const vertexSelected = isSelectedVertex('terminal', term.id, i);
        ctx.beginPath();
        ctx.fillStyle = vertexSelected ? '#f43f5e' : (i === 0 ? '#f97316' : '#e5e7eb');
        ctx.arc(x, y, layoutTerminalVertexRadiusPx(vertexSelected), 0, Math.PI*2);
        ctx.fill();
        if (vertexSelected) {
          ctx.strokeStyle = '#ffffff';
          ctx.lineWidth = 1.5;
          ctx.stroke();
        }
      });
      if (isDrawingTerm && state.layoutPathDrawPointer && term.vertices.length >= 1) {
        const ptr = state.layoutPathDrawPointer;
        const lastV = term.vertices[term.vertices.length - 1];
        const [lx, ly] = cellToPixel(lastV.col, lastV.row);
        if (ptr && ptr.length >= 2 && dist2([lx, ly], ptr) > 1e-6) {
          ctx.save();
          ctx.strokeStyle = 'rgba(250, 204, 21, 0.75)';
          ctx.setLineDash([4, 6]);
          ctx.lineWidth = 2;
          ctx.lineCap = 'round';
          ctx.beginPath();
          ctx.moveTo(lx, ly);
          ctx.lineTo(ptr[0], ptr[1]);
          ctx.stroke();
          ctx.restore();
        }
      }
    });
    ctx.restore();
  }

  function drawPBBs() {
    ctx.save();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.translate(state.panX, state.panY);
    ctx.scale(state.scale, state.scale);
    state.pbbStands.forEach(pbb => {
      const x1 = Number(pbb.x1), y1 = Number(pbb.y1), x2 = Number(pbb.x2), y2 = Number(pbb.y2);
      if (!Number.isFinite(x1) || !Number.isFinite(y1) || !Number.isFinite(x2) || !Number.isFinite(y2)) return;
      rebuildPbbBridgeGeometry(pbb);
      const endSize = getStandSizeMeters(pbb.category || 'C');
      const sel = state.selectedObject && state.selectedObject.type === 'pbb' && state.selectedObject.id === pbb.id;
      const simOcc = state.hasSimulationResult && isStandOccupiedAtSimSec(pbb.id, state.simTimeSec);
      const bridges = Array.isArray(pbb.pbbBridges) ? pbb.pbbBridges : [];
      bridges.forEach(function(bridge, bridgeIdx) {
        const pts = Array.isArray(bridge.points) ? bridge.points : [];
        if (pts.length < 2) return;
        ctx.strokeStyle = sel ? c2dObjectSelectedStroke() : '#f97316';
        ctx.lineWidth = sel ? 3.5 : 2.5;
        if (sel) {
          ctx.save();
          ctx.shadowColor = c2dObjectSelectedGlow();
          ctx.shadowBlur = c2dObjectSelectedGlowBlur();
        }
        ctx.beginPath();
        ctx.moveTo(Number(pts[0].x) || 0, Number(pts[0].y) || 0);
        for (let pi = 1; pi < pts.length; pi++) ctx.lineTo(Number(pts[pi].x) || 0, Number(pts[pi].y) || 0);
        ctx.stroke();
        if (sel) ctx.restore();
        if (sel) {
          pts.forEach(function(pt, ptIdx) {
            const isBridgeVertexSelected = !!(state.selectedVertex && state.selectedVertex.type === 'pbbBridge' && state.selectedVertex.id === pbb.id && state.selectedVertex.bridgeIndex === bridgeIdx && state.selectedVertex.pointIndex === ptIdx);
            ctx.beginPath();
            ctx.fillStyle = isBridgeVertexSelected ? '#f43f5e' : '#fdba74';
            ctx.arc(Number(pt.x) || 0, Number(pt.y) || 0, isBridgeVertexSelected ? 4 : 3, 0, Math.PI * 2);
            ctx.fill();
          });
        }
      });
      const apronPt = getStandConnectionPx(pbb);
      const ex = apronPt[0], ey = apronPt[1];
      const angle = getPBBStandAngle(pbb);
      const rotationActive = !!(state.selectedVertex && state.selectedVertex.type === 'standRotation' && state.selectedVertex.id === pbb.id);
      ctx.fillStyle = sel ? c2dObjectSelectedFill() : (simOcc ? c2dSimStandOccupiedFill() : 'rgba(22,163,74,0.18)');
      ctx.strokeStyle = sel ? c2dObjectSelectedStroke() : (simOcc ? c2dSimStandOccupiedStroke() : '#22c55e');
      ctx.lineWidth = sel ? 2.5 : 1.5;
      ctx.save();
      ctx.translate(ex, ey);
      ctx.rotate(angle);
      ctx.beginPath();
      ctx.rect(-endSize/2, -endSize/2, endSize, endSize);
      ctx.fill();
      if (sel) {
        ctx.save();
        ctx.shadowColor = c2dObjectSelectedGlow();
        ctx.shadowBlur = c2dObjectSelectedGlowBlur();
      }
      ctx.stroke();
      if (sel) ctx.restore();
      const nameRaw = (pbb.name && pbb.name.trim()) ? pbb.name.trim() : String(state.pbbStands.indexOf(pbb) + 1);
      const labelPrefix = getStandCategoryMode(pbb) === 'aircraft' ? 'AC' : (pbb.category || 'C');
      const label = labelPrefix + ' / ' + nameRaw;
      const pad = 3;
      const tx = endSize / 2 - pad;
      const ty = -endSize / 2 + pad;
      ctx.fillStyle = '#bbf7d0';
      ctx.font = '8px system-ui';
      ctx.textAlign = 'right';
      ctx.textBaseline = 'top';
      ctx.fillText(String(label), tx, ty);
      ctx.restore();
      ctx.save();
      ctx.beginPath();
      ctx.fillStyle = sel ? '#22c55e' : 'rgba(34,197,94,0.9)';
      ctx.arc(apronPt[0], apronPt[1], sel ? 4.5 : 3.5, 0, Math.PI * 2);
      ctx.fill();
      if (sel) {
        ctx.strokeStyle = '#bbf7d0';
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }
      ctx.restore();
      if (sel) {
        drawStandRotationHandle(getPbbRotationOriginPx(pbb), getPbbRotationHandlePx(pbb), rotationActive);
      }
    });
    ctx.restore();
  }

  function drawRemoteStands() {
    ctx.save();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.translate(state.panX, state.panY);
    ctx.scale(state.scale, state.scale);
    const mode = settingModeSelect ? settingModeSelect.value : 'grid';
    state.remoteStands.forEach(st => {
      const [cx, cy] = getRemoteStandCenterPx(st);
      const size = getStandSizeMeters(st.category || 'C');
      const angle = getRemoteStandAngleRad(st);
      const sel = state.selectedObject && state.selectedObject.type === 'remote' && state.selectedObject.id === st.id;
      const simOcc = state.hasSimulationResult && isStandOccupiedAtSimSec(st.id, state.simTimeSec);
      const rotationActive = !!(state.selectedVertex && state.selectedVertex.type === 'standRotation' && state.selectedVertex.id === st.id);
      ctx.save();
      ctx.translate(cx, cy);
      ctx.rotate(angle);
      ctx.fillStyle = sel ? c2dObjectSelectedFill() : (simOcc ? c2dSimStandOccupiedFill() : 'rgba(22,163,74,0.18)');
      ctx.strokeStyle = sel ? c2dObjectSelectedStroke() : (simOcc ? c2dSimStandOccupiedStroke() : '#22c55e');
      ctx.lineWidth = sel ? 2.5 : 1.5;
      ctx.beginPath();
      ctx.rect(-size/2, -size/2, size, size);
      ctx.fill();
      if (sel) {
        ctx.save();
        ctx.shadowColor = c2dObjectSelectedGlow();
        ctx.shadowBlur = c2dObjectSelectedGlowBlur();
      }
      ctx.stroke();
      if (sel) ctx.restore();
      ctx.restore();
      if (mode === 'apronTaxiway') {
        ctx.save();
        ctx.fillStyle = sel ? '#f97316' : '#e5e7eb';
        ctx.beginPath();
        ctx.arc(cx, cy, 2.5 * LAYOUT_VERTEX_DOT_SCALE, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
      }
      const nameRaw = (st.name && st.name.trim()) ? st.name.trim() : ('R' + String(state.remoteStands.indexOf(st) + 1).padStart(3, '0'));
      const labelPrefix = getStandCategoryMode(st) === 'aircraft' ? 'AC' : (st.category || 'C');
      const label = labelPrefix + ' / ' + nameRaw;
      ctx.fillStyle = '#bbf7d0';
      ctx.font = '8px system-ui';
      ctx.textAlign = 'right';
      ctx.textBaseline = 'top';
      const labelOffset = 2;
      ctx.fillText(label, cx + size/2 - labelOffset, cy - size/2 + labelOffset);
      if (sel) {
        drawStandRotationHandle([cx, cy], getRemoteRotationHandlePx(st), rotationActive);
      }
    });
    ctx.restore();
  }

  function renderRunwaySeparation() {
    const panel = document.getElementById('rwySepPanel');
    if (!panel) return;
    const runways = (state.taxiways || []).filter(tw => tw.pathType === 'runway');
    if (!runways.length) {
      panel.innerHTML = '<div style="font-size:11px;color:#9ca3af;">No runway paths. Layout Mode <strong>Runway</strong>Draw the runway polyline first with.</div>';
      return;
    }
    if (!state.activeRwySepId || !runways.some(r => r.id === state.activeRwySepId)) {
      state.activeRwySepId = runways[0].id;
    }
    const active = runways.find(r => r.id === state.activeRwySepId) || runways[0];
    const cfg = rsepGetConfigForRunway(active);
    const stdKey = cfg.standard || 'ICAO';
    const cats = RSEP_STD_CATS[stdKey] || RSEP_STD_CATS['ICAO'];
    const mode = cfg.mode || 'MIX';
    const seq = cfg.activeSeq || (RSEP_MODE_SEQS[mode] && RSEP_MODE_SEQS[mode][0]) || 'ARR→ARR';
    const seqType = RSEP_SEQ_TYPES[seq] || 'matrix';
    const seqMeta = rsepGetSeqMeta(seq);

    let html = '';
    html += '<div class="rwysep-rwy-bar">';
    html += '<div class="rwysep-rwy-tabs">';
    runways.forEach(rw => {
      const isActive = rw.id === active.id;
      const label = escapeHtml(rw.name || ('Runway ' + rw.id));
      html += '<button type="button" class="rwysep-rwy-btn' + (isActive ? ' active' : '') + '" data-rwy-id="' + rw.id + '">' + label + '</button>';
    });
    html += '</div></div>';

    const activeSub = 'noname';
    html += '<div class="layout-save-load-tabs" style="margin-top:8px;">';
    html += '<button type="button" class="layout-save-load-tab rwysep-subtab-btn active" data-subtab="noname">No Name</button>';
    html += '</div>';

    html += '<div id="rwysep-subtab-input" style="">';
    html += '<div class="rwysep-block">';
    html += '<div class="rwysep-label">Standard &amp; Mode</div>';
    html += '<div class="rwysep-row">';
    html += '<label style="font-size:11px;color:#9ca3af;">Standard&nbsp;</label>';
    html += '<select id="rwysep-standard">';
    html += '<option value="ICAO"' + (stdKey === 'ICAO' ? ' selected' : '') + '>ICAO (J/H/M/L)</option>';
    html += '<option value="RECAT-EU"' + (stdKey === 'RECAT-EU' ? ' selected' : '') + '>RECAT-EU (A~F)</option>';
    html += '</select>';
    html += '<label style="font-size:11px;color:#9ca3af;margin-left:8px;">Mode&nbsp;</label>';
    html += '<select id="rwysep-mode">';
    ['ARR','DEP','MIX'].forEach(m => {
      const txt = m === 'ARR' ? 'Arrivals only' : (m === 'DEP' ? 'Departures only' : 'Mixed (Arr/Dep)');
      html += '<option value="' + m + '"' + (mode === m ? ' selected' : '') + '>' + txt + '</option>';
    });
    html += '</select>';
    html += '<label style="font-size:11px;color:#9ca3af;margin-left:8px;">Seq&nbsp;</label>';
    html += '<select id="rwysep-seq">';
    (RSEP_MODE_SEQS[mode] || []).forEach(s => {
      const lbl = s;
      html += '<option value="' + s + '"' + (seq === s ? ' selected' : '') + '>' + lbl + '</option>';
    });
    html += '</select>';
    html += '</div></div>';

    if (seqMeta) {
      html += '<div class="rwysep-block" style="margin-top:4px;">';
      html += '<div class="rwysep-label">Concept summary</div>';
      html += '<div style="font-size:10px;color:#d1d5db;line-height:1.5;background:#020617;border-radius:6px;border:1px solid #111827;padding:6px 8px;">';
      html += '<div><span style="color:#9ca3af;">Driving factor</span>&nbsp;&nbsp;: ' + escapeHtml(seqMeta.driver) + '</div>';
      html += '<div><span style="color:#9ca3af;">Reference point</span>&nbsp;: ' + escapeHtml(seqMeta.refPoint) + '</div>';
      html += '<div><span style="color:#9ca3af;">Input structure</span>: ' + escapeHtml(seqMeta.input) + '</div>';
      html += '</div>';
      html += '</div>';
    }

    if (seq === 'ARR→DEP') {
      html += '<div class="rwysep-block">';
      html += '<div class="rwysep-label">ROT (Runway Occupancy Time, sec)</div>';

      const totalRot = cats.length;
      let filledRot = 0;
      cats.forEach(c => {
        const val = cfg.rot && cfg.rot[c] != null ? cfg.rot[c] : '';
        if (val !== '' && val != null) filledRot += 1;
      });
      html += rsepLegendHtml(filledRot, totalRot);

      html += '<div class="rwysep-row" style="flex-wrap:wrap;">';
      cats.forEach(c => {
        const rawVal = cfg.rot && cfg.rot[c] != null ? cfg.rot[c] : '';
        const valStr = rawVal === null || rawVal === undefined ? '' : String(rawVal);
        const sub = rsepGetCatLabel(stdKey, c);
        const colInfo = rsepColorForValue(valStr);
        html += '<div style="min-width:90px;margin-right:6px;margin-bottom:4px;">';
        html += '<div style="font-size:10px;color:#9ca3af;margin-bottom:2px;line-height:1.2;">';
        html += 'Cat ' + c;
        if (sub) {
          html += '<div style="font-size:9px;color:#6b7280;margin-top:1px;">' + escapeHtml(sub) + '</div>';
        }
        html += '</div>';
        html += '<input type="number" min="0" step="5" data-rwysep-rot="' + c + '" value="' + escapeHtml(valStr) + '" style="width:64px;background:' + colInfo.bg + ';border:1px solid ' + colInfo.border + ';color:' + colInfo.color + ';font-size:10px;padding:3px 4px;border-radius:3px;text-align:center;" />';
        html += ' <span style="font-size:9px;color:#6b7280;">sec</span>';
        html += '</div>';
      });
      html += '</div></div>';
    }

    if (seq === 'ARR→DEP') {
      html += '<div class="rwysep-block">';
      html += '<div class="rwysep-label">ROT‑based separation (sec)</div>';
      html += '<div style="font-size:10px;color:#9ca3af;line-height:1.5;">For ARR→DEP combinations, separation is determined by the ROT values above (runway occupancy time per wake category).</div>';
      html += '</div>';
    } else {
      html += '<div class="rwysep-block">';
      html += '<div class="rwysep-label">Separation (sec) — ' + escapeHtml(seq) + '</div>';
      if (seqType === 'matrix') {
        const data = cfg.seqData && cfg.seqData[seq] ? cfg.seqData[seq] : rsepMakeMatrix(cats, null);
        const total = cats.length * cats.length;
        let filled = 0;
        cats.forEach(lead => {
          cats.forEach(trail => {
            const v = data[lead] && data[lead][trail] != null ? data[lead][trail] : '';
            if (v !== '' && v != null) filled += 1;
          });
        });
        html += rsepLegendHtml(filled, total);

        html += '<div class="rwysep-matrix-wrap"><table class="rwysep-table"><thead><tr>';
        html += '<th>Lead↓ / Trail→</th>';
        cats.forEach(c => {
          const sub = rsepGetCatLabel(stdKey, c);
          html += '<th><div style="line-height:1.2;">' + c;
          if (sub) {
            html += '<div style="font-size:9px;color:#9ca3af;margin-top:1px;">' + escapeHtml(sub) + '</div>';
          }
          html += '</div></th>';
        });
        html += '</tr></thead><tbody>';
        cats.forEach(lead => {
          const leadSub = rsepGetCatLabel(stdKey, lead);
          html += '<tr><td><div style="line-height:1.2;">' + lead;
          if (leadSub) {
            html += '<div style="font-size:9px;color:#9ca3af;margin-top:1px;">' + escapeHtml(leadSub) + '</div>';
          }
          html += '</div></td>';
          cats.forEach(trail => {
            const v = data[lead] && data[lead][trail] != null ? data[lead][trail] : '';
            const colInfo = rsepColorForValue(v);
            html += '<td><input type="number" min="0" step="5" data-rwysep-matrix-lead="' + lead + '" data-rwysep-matrix-trail="' + trail + '" value="' + escapeHtml(String(v)) + '" style="width:64px;background:' + colInfo.bg + ';border:1px solid ' + colInfo.border + ';color:' + colInfo.color + ';font-size:10px;padding:3px 4px;border-radius:3px;text-align:center;" /></td>';
          });
          html += '</tr>';
        });
        html += '</tbody></table></div>';
      } else {
        const data1d = cfg.seqData && cfg.seqData[seq] ? cfg.seqData[seq] : rsepMake1D(cats, null);
        const total = cats.length;
        let filled = 0;
        cats.forEach(cat => {
          const v = data1d[cat] != null ? data1d[cat] : '';
          if (v !== '' && v != null) filled += 1;
        });
        html += rsepLegendHtml(filled, total);

        html += '<div class="rwysep-row" style="flex-wrap:wrap;margin-top:4px;">';
        cats.forEach(cat => {
          const v = data1d[cat] != null ? data1d[cat] : '';
          const colInfo = rsepColorForValue(v);
          const sub = rsepGetCatLabel(stdKey, cat);
          html += '<div style="min-width:90px;margin-right:6px;margin-bottom:4px;border:1px solid #1f2937;border-radius:6px;padding:6px 8px;background:#020617;">';
          html += '<div style="font-size:10px;color:#9ca3af;margin-bottom:2px;line-height:1.2;">Cat ' + cat;
          if (sub) {
            html += '<div style="font-size:9px;color:#6b7280;margin-top:1px;">' + escapeHtml(sub) + '</div>';
          }
          html += '</div>';
          html += '<input type="number" min="0" step="5" data-rwysep-1d="' + cat + '" value="' + escapeHtml(String(v)) + '" style="width:64px;background:' + colInfo.bg + ';border:1px solid ' + colInfo.border + ';color:' + colInfo.color + ';font-size:10px;padding:3px 4px;border-radius:3px;text-align:center;" />';
          html += ' <span style="font-size:9px;color:#6b7280;">sec</span>';
          html += '</div>';
        });
        html += '</div>';
      }
      html += '</div>';
    }
    html += '</div>'; // end subtab input

    html += '<div id="rwysep-subtab-timeline" style="' + (activeSub === 'timeline' ? '' : 'display:none;') + '">';
    html += '<div class="rwysep-block" style="margin-top:8px;">';
    html += '<div class="rwysep-label">Separation Timeline (Reg × Time)</div>';
    html += '<div id="rwySepTimeWrap" style="width:100%;background:#020617;border-radius:8px;border:1px solid #1f2937;position:relative;overflow-x:auto;overflow-y:auto;margin-top:4px;max-height:calc(40px * 12 + 80px);"></div>';
    html += '<div style="font-size:9px;color:#9ca3af;margin-top:4px;">';
    html += 'Y: Reg Number · X: Time · Bars = S-series (SLDT–STOT) · Lines = E-series (ELDT–ETOT)';
    html += '</div></div>';
    html += '</div>'; // end subtab timeline

    panel.innerHTML = html;

    function drawRwySeparationTimeline() {
      if (state.activeRwySepSubtab && state.activeRwySepSubtab !== 'timeline') return;
      const wrap = panel.querySelector('#rwySepTimeWrap');
      if (!wrap) return;

      const allData = typeof buildRunwaySeparationTimelineByRunwaySnapshot === 'function'
        ? buildRunwaySeparationTimelineByRunwaySnapshot(state.flights)
        : null;
      const data = allData && active && active.id != null ? allData[active.id] : null;
      if (!data || !data.events || !data.events.length) {
        wrap.innerHTML = '<div style="font-size:11px;color:#9ca3af;padding:8px 10px;">No SLDT/STOT events for this runway.</div>';
        return;
      }

      const byFlight = new Map();
      data.events.forEach(ev => {
        const f = ev.flight;
        if (!f) return;
        let lane = byFlight.get(f);
        if (!lane) {
          const reg = f.reg || f.id || '';
          lane = {
            flight: f,
            reg,
            hasArr: false,
            hasDep: false,
            sldt: null,
            eldt: null,
            stot: null,
            etot: null
          };
          byFlight.set(f, lane);
        }
        if (ev.type === 'arr') {
          lane.hasArr = true;
          lane.sldt = ev.time;
          lane.eldt = (f.eldtMin != null ? f.eldtMin : ev.time);
        } else if (ev.type === 'dep') {
          lane.hasDep = true;
          lane.stot = ev.time;
          lane.etot = (f.etotMin != null ? f.etotMin : ev.time);
        }
      });

      const lanes = Array.from(byFlight.values());
      if (!lanes.length) {
        wrap.innerHTML = '<div style="font-size:11px;color:#9ca3af;padding:8px 10px;">No SLDT/STOT events for this runway.</div>';
        return;
      }

      let minT0 = Infinity;
      let maxT0 = -Infinity;
      lanes.forEach(ln => {
        if (ln.sldt != null && ln.sldt < minT0) minT0 = ln.sldt;
        if (ln.etot != null && ln.etot > maxT0) maxT0 = ln.etot;
      });
      if (minT0 <= 0 && lanes.length) {
        const pos = lanes.map(function(ln) { return ln.sldt; }).filter(function(v) { return v != null && isFinite(v) && v > 1e-6; });
        if (pos.length) minT0 = Math.min.apply(null, pos);
      }
      if (!isFinite(minT0) || !isFinite(maxT0)) {
        minT0 = data.minT;
        maxT0 = data.maxT;
      }
      let baseMinT = Math.max(0, minT0 - RWY_SEP_TIMELINE_PAD_MIN);
      let baseMaxT = maxT0 + RWY_SEP_TIMELINE_PAD_MIN;
      if (baseMaxT <= baseMinT) baseMaxT = baseMinT + 60;
      const baseSpan = baseMaxT - baseMinT;
      const zoom = (state.rwySepTimeZoom && state.rwySepTimeZoom > 1) ? state.rwySepTimeZoom : 1;
      const span = baseSpan;
      const minT = baseMinT;
      const maxT = baseMaxT;

      lanes.sort((a, b) => {
        const ta = (a.sldt ?? a.stot ?? a.eldt ?? a.etot ?? 0);
        const tb = (b.sldt ?? b.stot ?? b.eldt ?? b.etot ?? 0);
        return ta - tb;
      });

      const tickPositions = buildTimeAxisTicks(minT, maxT, baseMinT, baseSpan, zoom);

      const sMarkers = [];
      const eMarkers = [];

      const rows = [];
      lanes.forEach(ln => {
        const reg = ln.reg || '';
        const sStart = (ln.sldt != null ? ln.sldt : null);
        const sEnd = (ln.stot != null ? ln.stot : null);
        const eStart = (ln.eldt != null ? ln.eldt : null);
        const eEnd = (ln.etot != null ? ln.etot : null);

        let blocks = '';
        if (sStart != null && sEnd != null && span > 0) {
          const s1 = Math.max(sStart, baseMinT);
          const s2 = Math.min(sEnd, baseMaxT);
          if (s2 <= s1) return;
          const leftPct = ((s1 - baseMinT) / baseSpan) * 100 * zoom;
          const widthPct = Math.max(1, ((s2 - s1) / baseSpan) * 100 * zoom);
          const rightPct = leftPct + widthPct;
          sMarkers.push({ t: sStart, leftPct, type: 'start' });
          sMarkers.push({ t: sEnd, leftPct: rightPct, type: 'end' });
          blocks +=
            '<div class="rwysep-line-s" style="' +
              'left:' + leftPct + '%;' +
              'width:' + widthPct + '%;' +
            '"></div>' +
            '<div class="rwysep-tri" style="' +
              'top:20%;' +
              'left:' + leftPct + '%;' +
              'border-top:6px solid ' + GANTT_COLORS.S_SERIES + ';' +
            '"></div>' +
            '<div class="rwysep-tri" style="' +
              'top:20%;' +
              'left:' + rightPct + '%;' +
              'border-bottom:6px solid ' + GANTT_COLORS.S_SERIES + ';' +
            '"></div>';
        }
        if (eStart != null && eEnd != null && span > 0) {
          const e1 = Math.max(eStart, baseMinT);
          const e2 = Math.min(eEnd, baseMaxT);
          if (e2 <= e1) return;
          const leftPct2 = ((e1 - baseMinT) / baseSpan) * 100 * zoom;
          const widthPct2 = Math.max(0.5, ((e2 - e1) / baseSpan) * 100 * zoom);
          const rightPct2 = leftPct2 + widthPct2;
          eMarkers.push({ t: eStart, leftPct: leftPct2, type: 'start' });
          eMarkers.push({ t: eEnd, leftPct: rightPct2, type: 'end' });
          blocks +=
            '<div class="rwysep-line-e" style="' +
              'left:' + leftPct2 + '%;' +
              'width:' + widthPct2 + '%;' +
            '"></div>' +
            '<div class="rwysep-tri" style="' +
              'top:54%;' +
              'left:' + leftPct2 + '%;' +
              'border-top:6px solid ' + GANTT_COLORS.E_SERIES + ';' +
            '"></div>' +
            '<div class="rwysep-tri" style="' +
              'top:54%;' +
              'left:' + rightPct2 + '%;' +
              'border-bottom:6px solid ' + GANTT_COLORS.E_SERIES + ';' +
            '"></div>';
        }

        rows.push(
          '<div class="alloc-row">' +
            '<div class="alloc-row-label">' + escapeHtml(reg) + '</div>' +
            '<div class="alloc-row-track" style="background:transparent;border:none;">' + blocks + '</div>' +
          '</div>'
        );
      });

      sMarkers.sort((a, b) => a.t - b.t);
      eMarkers.sort((a, b) => a.t - b.t);

      const sHeadMarks = sMarkers.map(m =>
        '<div class="rwysep-tri" style="' +
          'top:60%;' +
          'left:' + m.leftPct + '%;' +
          (m.type === 'start'
            ? 'border-top:6px solid ' + GANTT_COLORS.S_SERIES + ';'
            : 'border-bottom:6px solid ' + GANTT_COLORS.S_SERIES + ';') +
        '"></div>'
      ).join('');

      const eHeadMarks = eMarkers.map(m =>
        '<div class="rwysep-tri" style="' +
          'top:60%;' +
          'left:' + m.leftPct + '%;' +
          (m.type === 'start'
            ? 'border-top:6px solid ' + GANTT_COLORS.E_SERIES + ';'
            : 'border-bottom:6px solid ' + GANTT_COLORS.E_SERIES + ';') +
        '"></div>'
      ).join('');

      const headHtml =
        '<div class="rwysep-head-row">' +
          '<div class="rwysep-head-label">S-series</div>' +
          '<div class="rwysep-head-track">' + sHeadMarks + '</div>' +
        '</div>' +
        '<div class="rwysep-head-row">' +
          '<div class="rwysep-head-label">E-series</div>' +
          '<div class="rwysep-head-track">' + eHeadMarks + '</div>' +
        '</div>';

      const axisTicks = tickPositions.map(tp =>
        '<div class="alloc-time-tick" style="left:' + tp.leftPct + '%;">' +
          '<div class="alloc-time-tick-label">' + tp.label + '</div>' +
        '</div>'
      );
      const axisHtml =
        '<div class="alloc-time-axis-overlay">' +
          '<div class="alloc-time-axis-inner">' + axisTicks.join('') + '</div>' +
        '</div>';

      const rwyGridOverlay =
        '<div class="alloc-gantt-grid-overlay">' +
          tickPositions.map(function(tp) {
            return '<div class="alloc-time-grid-line" style="left:' + tp.leftPct + '%;"></div>';
          }).join('') +
        '</div>';
      const rowsHtml = '<div class="rwysep-rows">' + rwyGridOverlay + rows.join('') + '</div>';
      wrap.innerHTML = headHtml + rowsHtml + axisHtml;

      if (!wrap._rwySepZoomBound) {
        wrap._rwySepZoomBound = true;
        wrap.addEventListener('wheel', function(e) {
          if (!e.shiftKey) return;
          e.preventDefault();
          const factor = e.deltaY < 0 ? 1.15 : (1 / 1.15);
          let z = state.rwySepTimeZoom || 1;
          z *= factor;
          if (z < 1) z = 1;
          if (z > 8) z = 8;
          state.rwySepTimeZoom = z;
          if (typeof renderRunwaySeparation === 'function') renderRunwaySeparation();
        }, { passive: false });
      }

      if (!wrap._rwySepScrollBound) {
        wrap._rwySepScrollBound = true;
        wrap.addEventListener('scroll', function() {
          if (wrap._rwySepScrollRecalc) return;
          const currentLeft = wrap.scrollLeft;
          wrap._rwySepScrollRecalc = true;
          drawRwySeparationTimeline();
          wrap.scrollLeft = currentLeft;
          wrap._rwySepScrollRecalc = false;
        });
      }
    }

    drawRwySeparationTimeline();

    _rwySepWireInputHandlers(panel, cfg, cats, seq, state);
  }

  function _rwySepWireInputHandlers(panel, cfg, cats, seq, st) {
    panel.querySelectorAll('.rwysep-rwy-btn').forEach(function(btn) {
      btn.addEventListener('click', function() {
        const id = this.getAttribute('data-rwy-id');
        if (!id) return;
        st.activeRwySepId = id;
        renderRunwaySeparation();
      });
    });
    panel.querySelectorAll('.rwysep-subtab-btn').forEach(function(btn) {
      btn.addEventListener('click', function() {
        const sub = this.getAttribute('data-subtab') || 'input';
        st.activeRwySepSubtab = sub;
        renderRunwaySeparation();
      });
    });
    var stdSel = panel.querySelector('#rwysep-standard');
    if (stdSel) {
      stdSel.addEventListener('change', function() {
        cfg.standard = this.value || 'ICAO';
        cfg.seqData = rsepMakeSeqData(cfg.standard);
        var catsNew = RSEP_STD_CATS[cfg.standard] || [];
        var rotNew = RSEP_STANDARDS[cfg.standard] && RSEP_STANDARDS[cfg.standard].ROT || {};
        cfg.rot = {};
        catsNew.forEach(function(c) { cfg.rot[c] = rotNew[c] != null ? String(rotNew[c]) : ''; });
        renderRunwaySeparation();
      });
    }
    var modeSel = panel.querySelector('#rwysep-mode');
    if (modeSel) {
      modeSel.addEventListener('change', function() {
        cfg.mode = this.value || 'MIX';
        var seqs = RSEP_MODE_SEQS[cfg.mode] || ['ARR→ARR'];
        if (!seqs.includes(cfg.activeSeq)) cfg.activeSeq = seqs[0];
        renderRunwaySeparation();
      });
    }
    var seqSel = panel.querySelector('#rwysep-seq');
    if (seqSel) {
      seqSel.addEventListener('change', function() {
        cfg.activeSeq = this.value || 'ARR→ARR';
        renderRunwaySeparation();
      });
    }
    function _applyColorOnChange(inp) {
      var colInfo = rsepColorForValue(inp.value);
      inp.style.background = colInfo.bg;
      inp.style.borderColor = colInfo.border;
      inp.style.color = colInfo.color;
    }
    panel.querySelectorAll('input[data-rwysep-rot]').forEach(function(inp) {
      inp.addEventListener('change', function() {
        var cat = this.getAttribute('data-rwysep-rot');
        if (!cat) return;
        cfg.rot[cat] = this.value;
        _applyColorOnChange(this);
      });
    });
    panel.querySelectorAll('input[data-rwysep-matrix-lead]').forEach(function(inp) {
      inp.addEventListener('change', function() {
        var lead = this.getAttribute('data-rwysep-matrix-lead');
        var trail = this.getAttribute('data-rwysep-matrix-trail');
        if (!lead || !trail) return;
        if (!cfg.seqData[seq]) cfg.seqData[seq] = rsepMakeMatrix(cats, null);
        if (!cfg.seqData[seq][lead]) cfg.seqData[seq][lead] = {};
        cfg.seqData[seq][lead][trail] = this.value;
        _applyColorOnChange(this);
      });
    });
    panel.querySelectorAll('input[data-rwysep-1d]').forEach(function(inp) {
      inp.addEventListener('change', function() {
        var cat = this.getAttribute('data-rwysep-1d');
        if (!cat) return;
        if (!cfg.seqData[seq]) cfg.seqData[seq] = rsepMake1D(cats, null);
        cfg.seqData[seq][cat] = this.value;
        _applyColorOnChange(this);
      });
    });
  }

  function drawTaxiways() {
    ctx.save();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.translate(state.panX, state.panY);
    ctx.scale(state.scale, state.scale);
    state.taxiways.forEach(tw => {
      const drawing = state.taxiwayDrawingId === tw.id;
      if (tw.vertices.length < 2 && !drawing) return;
      const isRunwayPath = tw.pathType === 'runway';
      const isRunwayExit = tw.pathType === 'runway_exit';
      const widthDefault = isRunwayPath ? RUNWAY_PATH_DEFAULT_WIDTH : (isRunwayExit ? RUNWAY_EXIT_DEFAULT_WIDTH : TAXIWAY_DEFAULT_WIDTH);
      const width = tw.width != null ? tw.width : widthDefault;
      const sel = state.selectedObject && state.selectedObject.type === 'taxiway' && state.selectedObject.id === tw.id;
      const pathLineCap = 'butt';
      if (sel) {
        ctx.strokeStyle = c2dObjectSelectedStroke();
        ctx.fillStyle = c2dObjectSelectedFill();
      } else if (isRunwayPath || isRunwayExit) {
        ctx.strokeStyle = c2dRunwayStroke();
        ctx.fillStyle = c2dRunwayFill();
      } else {
        ctx.strokeStyle = drawing ? 'rgba(250, 204, 21, 0.78)' : 'rgba(251, 191, 36, 0.72)';
        ctx.fillStyle = 'rgba(251,191,36,0.18)';
      }
      ctx.lineWidth = width;
      ctx.lineCap = pathLineCap;
      ctx.lineJoin = 'round';
      ctx.beginPath();
      for (let i = 0; i < tw.vertices.length; i++) {
        const [x, y] = cellToPixel(tw.vertices[i].col, tw.vertices[i].row);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      if (tw.vertices.length >= 2) {
        if (sel) {
          ctx.save();
          ctx.shadowColor = c2dObjectSelectedGlow();
          ctx.shadowBlur = c2dObjectSelectedGlowBlur();
          ctx.stroke();
          ctx.restore();
        } else ctx.stroke();
      }
      if (!isRunwayPath) {
        ctx.lineWidth = 1.5;
        ctx.strokeStyle = sel ? c2dObjectSelectedStroke() : (isRunwayExit ? c2dPassengerTerminalStroke() : '#facc15');
        ctx.beginPath();
        for (let i = 0; i < tw.vertices.length; i++) {
          const [x, y] = cellToPixel(tw.vertices[i].col, tw.vertices[i].row);
          if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        if (tw.vertices.length >= 2) ctx.stroke();
      }
      if (isRunwayPath && tw.vertices.length >= 2) {
        const runwayPts = tw.vertices.map(v => cellToPixel(v.col, v.row));
        drawRunwayDecorations(tw, runwayPts, width);
      }
      const dir = getTaxiwayDirection(tw);
      if (dir !== 'both' && tw.vertices.length >= 2) {
        const pts = tw.vertices.map(v => cellToPixel(v.col, v.row));
        const totalLen = pts.reduce((acc, p, i) => acc + (i > 0 ? Math.hypot(p[0]-pts[i-1][0], p[1]-pts[i-1][1]) : 0), 0);
        const arrowSpacing = Math.max(22, Math.min(42, totalLen / 10));
        const numArrows = Math.max(2, Math.floor(totalLen / arrowSpacing));
        const arrLen = CELL_SIZE * 0.54;
        ctx.fillStyle = '#f5930b';
        for (let k = 1; k <= numArrows; k++) {
          const targetDist = totalLen * (k / (numArrows + 1));
          let acc = 0;
          let ax = pts[0][0], ay = pts[0][1];
          let angle = Math.atan2(pts[1][1]-pts[0][1], pts[1][0]-pts[0][0]);
          for (let i = 1; i < pts.length; i++) {
            const seg = Math.hypot(pts[i][0]-pts[i-1][0], pts[i][1]-pts[i-1][1]);
            angle = Math.atan2(pts[i][1]-pts[i-1][1], pts[i][0]-pts[i-1][0]);
            if (acc + seg >= targetDist) {
              const t = seg > 0 ? (targetDist - acc) / seg : 0;
              ax = pts[i-1][0] + t * (pts[i][0]-pts[i-1][0]);
              ay = pts[i-1][1] + t * (pts[i][1]-pts[i-1][1]);
              break;
            }
            acc += seg;
          }
          if (dir === 'counter_clockwise') angle += Math.PI;
          ctx.beginPath();
          ctx.moveTo(ax + arrLen * Math.cos(angle), ay + arrLen * Math.sin(angle));
          ctx.lineTo(ax - arrLen * 0.7 * Math.cos(angle) + arrLen * 0.4 * Math.sin(angle), ay - arrLen * 0.7 * Math.sin(angle) - arrLen * 0.4 * Math.cos(angle));
          ctx.lineTo(ax - arrLen * 0.7 * Math.cos(angle) - arrLen * 0.4 * Math.sin(angle), ay - arrLen * 0.7 * Math.sin(angle) + arrLen * 0.4 * Math.cos(angle));
          ctx.closePath();
          ctx.fill();
        }
      }
      if (isRunwayPath && tw.vertices.length >= 2) {
        const rp = getRunwayPath(tw.id);
        if (rp && rp.pts.length >= 2) {
          const lenPx = runwayPolylineLengthPx(rp.pts);
          const d = Math.min(Math.max(0, getEffectiveRunwayLineupDistM(tw)), lenPx);
          const lp = getRunwayPointAtDistance(tw.id, d);
          if (lp) {
            const lineupRtxOk = isLineupPointTouchingRunwayTaxiwayOnRunway(tw, lp);
            ctx.save();
            ctx.fillStyle = lineupRtxOk ? '#16a34a' : '#dc2626';
            ctx.strokeStyle = lineupRtxOk ? '#14532d' : '#450a0a';
            ctx.lineWidth = 1.2;
            ctx.beginPath();
            ctx.arc(lp[0], lp[1], 5 * LAYOUT_VERTEX_DOT_SCALE, 0, Math.PI * 2);
            ctx.fill();
            ctx.stroke();
            const labelText = 'Line up';
            ctx.font = 'bold 11px system-ui, sans-serif';
            const padX = 6, padY = 4, rad = 5;
            const mLabel = ctx.measureText(labelText);
            const bw = mLabel.width + padX * 2;
            const bh = 11 + padY * 2;
            const bx = lp[0] + 7;
            const by = lp[1] - 4 - bh;
            ctx.beginPath();
            if (typeof ctx.roundRect === 'function') ctx.roundRect(bx, by, bw, bh, rad);
            else ctx.rect(bx, by, bw, bh);
            ctx.fillStyle = lineupRtxOk ? 'rgba(22, 163, 74, 0.92)' : 'rgba(220, 38, 38, 0.92)';
            ctx.fill();
            ctx.strokeStyle = lineupRtxOk ? '#14532d' : '#450a0a';
            ctx.lineWidth = 1.2;
            ctx.stroke();
            ctx.fillStyle = '#ffffff';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(labelText, bx + bw / 2, by + bh / 2);
            let hpLinked = false;
            if (typeof findRunwayHoldingForLineup === 'function' && lp) {
              const dLine = typeof getEffectiveRunwayLineupDistM === 'function' ? getEffectiveRunwayLineupDistM(tw) : d;
              const dBack = Math.max(0, dLine - 300);
              const pBack = typeof getRunwayPointAtDistance === 'function' ? getRunwayPointAtDistance(tw.id, dBack) : null;
              const stub = (pBack && dist2(pBack, lp) > 1e-6) ? [pBack, lp] : [lp, lp];
              const hres = findRunwayHoldingForLineup(tw, lp, stub);
              hpLinked = !!(hres && hres.ok);
            }
            if (!hpLinked) {
              const badge = 'No Holding Point';
              ctx.font = 'bold 10px system-ui, sans-serif';
              const p2x = 5, p2y = 3, r2 = 4;
              const m2 = ctx.measureText(badge);
              const bw2 = m2.width + p2x * 2;
              const bh2 = 10 + p2y * 2;
              const bx2 = bx;
              const by2 = by + bh + 4;
              ctx.beginPath();
              if (typeof ctx.roundRect === 'function') ctx.roundRect(bx2, by2, bw2, bh2, r2);
              else ctx.rect(bx2, by2, bw2, bh2);
              ctx.fillStyle = 'rgba(220, 38, 38, 0.95)';
              ctx.fill();
              ctx.strokeStyle = '#450a0a';
              ctx.lineWidth = 1.1;
              ctx.stroke();
              ctx.fillStyle = '#ffffff';
              ctx.textAlign = 'center';
              ctx.textBaseline = 'middle';
              ctx.fillText(badge, bx2 + bw2 / 2, by2 + bh2 / 2);
            }
            ctx.restore();
          }
        }
      }
      if ((drawing || sel) && tw.vertices.length >= 1) {
        tw.vertices.forEach((v, i) => {
          const [x, y] = cellToPixel(v.col, v.row);
          const vertexSelected = isSelectedVertex('taxiway', tw.id, i);
          if (i === 0 && drawing) {
            ctx.fillStyle = '#f97316';
            ctx.beginPath();
            ctx.arc(x, y, c2dPathDrawStartMarkerRadiusPx(), 0, Math.PI*2);
            ctx.fill();
            ctx.strokeStyle = '#ea580c';
            ctx.lineWidth = c2dPathDrawStartMarkerStrokePx();
            ctx.stroke();
            ctx.fillStyle = '#fff';
            ctx.font = 'bold ' + c2dPathDrawStartLabelFontPx() + 'px system-ui';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText('Start', x, y + c2dPathDrawStartLabelOffsetY());
          } else {
            ctx.fillStyle = vertexSelected ? '#f43f5e' : ((i === 0 && sel) ? '#f97316' : '#e5e7eb');
            ctx.beginPath();
            ctx.arc(x, y, layoutPathVertexRadiusPx(vertexSelected, sel), 0, Math.PI*2);
            ctx.fill();
            if (sel || vertexSelected) {
              ctx.strokeStyle = vertexSelected ? '#ffffff' : c2dObjectSelectedStroke();
              ctx.lineWidth = 1.5;
              ctx.stroke();
            }
          }
        });
      }
      if (drawing && state.layoutPathDrawPointer && tw.vertices.length >= 1) {
        const ptr = state.layoutPathDrawPointer;
        const lastV = tw.vertices[tw.vertices.length - 1];
        const [lx, ly] = cellToPixel(lastV.col, lastV.row);
        if (ptr && ptr.length >= 2 && dist2([lx, ly], ptr) > 1e-6) {
          ctx.save();
          ctx.strokeStyle = 'rgba(250, 204, 21, 0.75)';
          ctx.setLineDash([4, 6]);
          ctx.lineWidth = Math.max(2, width * 0.25);
          ctx.lineCap = 'round';
          ctx.beginPath();
          ctx.moveTo(lx, ly);
          ctx.lineTo(ptr[0], ptr[1]);
          ctx.stroke();
          ctx.restore();
        }
      }
    });
    ctx.restore();
  }

  function drawApronTaxiwayLinks() {
    ctx.save();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.translate(state.panX, state.panY);
    ctx.scale(state.scale, state.scale);
    ctx.lineWidth = 3;
    ctx.setLineDash([3, 3]);
    state.apronLinks.forEach(lk => {
      const stand = findStandById(lk.pbbId);
      const tw = state.taxiways.find(t => t.id === lk.taxiwayId);
      if (!stand || !tw || lk.tx == null || lk.ty == null) return;
      const poly = getApronLinkPolylineWorldPts(lk);
      if (poly.length < 2) return;
      ctx.strokeStyle = '#facc15';
      ctx.beginPath();
      ctx.moveTo(poly[0][0], poly[0][1]);
      for (let pi = 1; pi < poly.length; pi++) ctx.lineTo(poly[pi][0], poly[pi][1]);
      ctx.stroke();
      const svApron = state.selectedVertex;
      const selApron = state.selectedObject && state.selectedObject.type === 'apronLink' && state.selectedObject.id === lk.id;
      if (selApron) {
        ctx.setLineDash([]);
        for (let pi = 0; pi < poly.length; pi++) {
          const [px, py] = poly[pi];
          const isStandEnd = (pi === 0);
          const isTaxiEnd = (pi === poly.length - 1);
          const midIdx = isStandEnd || isTaxiEnd ? -1 : (pi - 1);
          let vtxSel = false;
          let draggable = false;
          if (isTaxiEnd) {
            draggable = true;
            vtxSel = !!(svApron && svApron.type === 'apronLink' && svApron.id === lk.id && svApron.kind === 'taxiway');
          } else if (!isStandEnd) {
            draggable = true;
            vtxSel = !!(svApron && svApron.type === 'apronLink' && svApron.id === lk.id && svApron.kind === 'mid' && svApron.midIndex === midIdx);
          }
          const r = layoutPathVertexRadiusPx(vtxSel, draggable);
          ctx.fillStyle = vtxSel ? '#f43f5e' : (draggable ? '#fde68a' : '#facc15');
          ctx.beginPath();
          ctx.arc(px, py, r, 0, Math.PI*2);
          ctx.fill();
          if (vtxSel || draggable) {
            ctx.strokeStyle = vtxSel ? '#ffffff' : c2dObjectSelectedStroke();
            ctx.lineWidth = 1.5;
            ctx.stroke();
          }
        }
        ctx.setLineDash([3, 3]);
      }
    });
    ctx.setLineDash([]);
    if (state.apronLinkTemp) {
      ctx.fillStyle = '#facc15';
      const t = state.apronLinkTemp;
      const draft = [];
      if (t.kind === 'pbb' || t.kind === 'remote') {
        const st = findStandById(t.standId);
        if (st) {
          draft.push(getStandConnectionPx(st));
        }
      } else if (t.kind === 'taxiway') {
        draft.push([t.x, t.y]);
      }
      (state.apronLinkMidpoints || []).forEach(function(c) {
        draft.push(cellToPixel(c.col, c.row));
      });
      if (state.apronLinkPointerWorld && state.apronLinkPointerWorld.length >= 2) draft.push(state.apronLinkPointerWorld);
      if (draft.length >= 1) {
        ctx.save();
        ctx.strokeStyle = 'rgba(250, 204, 21, 0.75)';
        ctx.setLineDash([4, 6]);
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(draft[0][0], draft[0][1]);
        for (let di = 1; di < draft.length; di++) ctx.lineTo(draft[di][0], draft[di][1]);
        if (draft.length >= 2) ctx.stroke();
        ctx.restore();
        draft.forEach(function(pt) {
          ctx.beginPath();
          ctx.arc(pt[0], pt[1], CELL_SIZE * 0.2 * LAYOUT_VERTEX_DOT_SCALE, 0, Math.PI*2);
          ctx.fill();
        });
      }
    }
    ctx.restore();
  }

  function drawHoldingPoints2D() {
    if (!ctx) return;
    ctx.save();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.translate(state.panX, state.panY);
    ctx.scale(state.scale, state.scale);
    const r = c2dHoldingPointDiameterM() * 0.5;
    const sel = state.selectedObject && state.selectedObject.type === 'holdingPoint';
    (state.holdingPoints || []).forEach(function(hp) {
      if (!hp || !isFinite(hp.x) || !isFinite(hp.y)) return;
      const selected = sel && state.selectedObject.id === hp.id;
      const k = normalizeHoldingPointKind(hp.hpKind);
      const fill = c2dHoldingPointFillForKind(k);
      const stroke = c2dHoldingPointStrokeForKind(k);
      ctx.beginPath();
      ctx.arc(hp.x, hp.y, r, 0, Math.PI * 2);
      ctx.fillStyle = selected ? c2dObjectSelectedFill() : fill;
      ctx.strokeStyle = selected ? c2dObjectSelectedStroke() : stroke;
      ctx.lineWidth = selected ? 2.5 : 1;
      if (selected) {
        ctx.shadowColor = c2dObjectSelectedGlow();
        ctx.shadowBlur = c2dObjectSelectedGlowBlur();
      } else {
        ctx.shadowBlur = 0;
      }
      ctx.fill();
      ctx.stroke();
      ctx.shadowBlur = 0;
    });
    if (state.holdingPointDrawing && state.previewHoldingPoint) {
      const px = state.previewHoldingPoint.x, py = state.previewHoldingPoint.y;
      const ptp = state.previewHoldingPoint.pathType || 'taxiway';
      ctx.beginPath();
      ctx.arc(px, py, r, 0, Math.PI * 2);
      ctx.fillStyle = c2dHoldingPointPreviewFillForPathType(ptp);
      ctx.strokeStyle = c2dHoldingPointPreviewStrokeForPathType(ptp);
      ctx.lineWidth = 1;
      ctx.shadowBlur = 0;
      ctx.fill();
      ctx.stroke();
    }
    ctx.restore();
  }

  function drawStandPreview() {
    ctx.save();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.translate(state.panX, state.panY);
    ctx.scale(state.scale, state.scale);
    const mode = settingModeSelect.value;
    if (mode === 'remote' && state.previewRemote) {
      const cx = Number(state.previewRemote.x), cy = Number(state.previewRemote.y);
      const category = document.getElementById('remoteCategory').value || 'C';
      const size = getStandSizeMeters(category);
      const angle = normalizeAngleDeg(document.getElementById('remoteAngle') ? document.getElementById('remoteAngle').value : 0) * Math.PI / 180;
      const overlap = state.previewRemote.overlap;
      ctx.fillStyle = overlap ? 'rgba(239,68,68,0.35)' : 'rgba(34,197,94,0.25)';
      ctx.strokeStyle = overlap ? '#ef4444' : '#22c55e';
      ctx.lineWidth = 2;
      ctx.setLineDash([4, 4]);
      ctx.save();
      ctx.translate(cx, cy);
      ctx.rotate(angle);
      ctx.beginPath();
      ctx.rect(-size/2, -size/2, size, size);
      ctx.fill();
      ctx.stroke();
      ctx.restore();
    }
    if (mode === 'pbb' && state.previewPbb) {
      const ex = state.previewPbb.x2, ey = state.previewPbb.y2;
      const size = getStandSizeMeters(state.previewPbb.category || 'C');
      const overlap = state.previewPbb.overlap;
      const angle = getPBBStandAngle(state.previewPbb);
      ctx.fillStyle = overlap ? 'rgba(239,68,68,0.35)' : 'rgba(34,197,94,0.25)';
      ctx.strokeStyle = overlap ? '#ef4444' : '#22c55e';
      ctx.lineWidth = 2;
      ctx.setLineDash([4, 4]);
      ctx.save();
      ctx.translate(ex, ey);
      ctx.rotate(angle);
      ctx.beginPath();
      ctx.rect(-size/2, -size/2, size, size);
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = '#bbf7d0';
      ctx.font = '10px system-ui';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(state.previewPbb.category || document.getElementById('standCategory').value || 'C', 0, 0);
      ctx.restore();
    }
    ctx.restore();
  }

  let _safeDrawErrLogged = false;
  let _drawRafId = 0;
  function safeDraw() { try { draw(); _safeDrawErrLogged = false; } catch(e) { if (!_safeDrawErrLogged) { console.error('safeDraw: draw() error', e); _safeDrawErrLogged = true; } } }
  function flushDrawNow() {
    if (_drawRafId) {
      cancelAnimationFrame(_drawRafId);
      _drawRafId = 0;
    }
    safeDraw();
  }
  function scheduleDraw() {
    if (_drawRafId) return;
    _drawRafId = requestAnimationFrame(function() {
      _drawRafId = 0;
      safeDraw();
    });
  }
  function draw() {
    if (!ctx || !canvas) return;
    drawGrid();
    drawTerminals();
    drawTaxiways();
    drawHoldingPoints2D();
    drawPBBs();
    drawRemoteStands();
    drawApronTaxiwayLinks();
    drawStandPreview();
    drawSelectedLayoutEdge();
    {
      const sel = state.selectedObject;
      const rid = state.flightPathRevealFlightId;
      if (sel && sel.type === 'flight' && rid != null && sel.id === rid) {
        drawFlightPathHighlight();
        drawDeparturePathHighlight();
      }
    }
    drawApproachPreviewPaths2D();
    drawFlights2D();
    drawPathJunctions();
  }

  document.addEventListener('keydown', function(ev) {
    const el = document.activeElement;
    const inInput = el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.isContentEditable);
    if (ev.ctrlKey && ev.key === 'z') {
      if (!inInput) { ev.preventDefault(); undo(); }
      return;
    }
    if (ev.key !== 'Delete' && ev.key !== 'Backspace') return;
    if (inInput) return;
    if (removeLastDrawingVertex()) {
      ev.preventDefault();
      return;
    }
    if (removeSelectedVertex()) {
      ev.preventDefault();
      return;
    }
    if (!state.selectedObject) return;
    const type = state.selectedObject.type;
    const id = state.selectedObject.id;
    if (type !== 'terminal' && type !== 'pbb' && type !== 'remote' && type !== 'holdingPoint' && type !== 'taxiway' && type !== 'apronLink' && type !== 'flight') return;
    pushUndo();
    removeLayoutObjectFromState(type, id);
    state.selectedObject = null;
    state.selectedVertex = null;
    if (type === 'terminal' && state.currentTerminalId === id) {
      state.currentTerminalId = state.terminals.length ? state.terminals[0].id : null;
      if (state.terminalDrawingId === id) {
        state.terminalDrawingId = null;
        state.layoutPathDrawPointer = null;
      }
    }
    if (type === 'taxiway' && state.taxiwayDrawingId === id) {
      state.taxiwayDrawingId = null;
      state.layoutPathDrawPointer = null;
    }
    syncPanelFromState();
    updateObjectInfo();
    if (typeof redrawLayoutAfterEdit === 'function') redrawLayoutAfterEdit();
    else if (typeof updateAllFlightPaths === 'function') updateAllFlightPaths(); else draw();
    ev.preventDefault();
  });

  container.addEventListener('mousedown', function(ev) {
    if (ev.button !== 0) return;
    const rect = canvas.getBoundingClientRect();
    const sx = ev.clientX - rect.left, sy = ev.clientY - rect.top;
    const [wx, wy] = screenToWorld(sx, sy);
    const mode = settingModeSelect.value;
    if (mode === 'terminal' && !state.terminalDrawingId) {
      const vhit = hitTestTerminalVertex(wx, wy);
      if (vhit) {
        pushUndo();
        state.dragVertex = vhit;
        state.selectedVertex = { type: 'terminal', id: vhit.terminalId, index: vhit.index };
        const term = state.terminals.find(t => t.id === vhit.terminalId);
        if (term) {
          state.flightPathRevealFlightId = null;
          state.selectedObject = { type: 'terminal', id: term.id, obj: term };
          state.currentTerminalId = term.id;
          syncPanelFromState();
          updateObjectInfo();
          draw();
        }
        return;
      }
    }
    if (state.selectedObject && state.selectedObject.type === 'taxiway') {
      const thit = hitTestTaxiwayVertex(wx, wy);
      if (thit && thit.taxiwayId === state.selectedObject.id) {
        pushUndo();
        state.dragTaxiwayVertex = thit;
        state.selectedVertex = { type: 'taxiway', id: thit.taxiwayId, index: thit.index };
        draw();
        return;
      }
    }
    const standRotateHit = hitTestStandRotationHandle(wx, wy);
    if (standRotateHit) {
      pushUndo();
      state.dragStandRotation = standRotateHit;
      state.selectedVertex = { type: 'standRotation', id: standRotateHit.id, standType: standRotateHit.type };
      draw();
      return;
    }
    if (state.selectedObject && state.selectedObject.type === 'pbb' && !state.pbbDrawing) {
      const ph = hitTestPbbEditablePoint(wx, wy);
      if (ph) {
        pushUndo();
        if (ph.type === 'bridge') {
          state.dragPbbBridgeVertex = { pbbId: state.selectedObject.id, bridgeIndex: ph.bridgeIndex, pointIndex: ph.pointIndex };
          state.selectedVertex = { type: 'pbbBridge', id: state.selectedObject.id, bridgeIndex: ph.bridgeIndex, pointIndex: ph.pointIndex };
        } else {
          state.dragStandConnection = { pbbId: state.selectedObject.id };
          state.selectedVertex = { type: 'pbbApronSite', id: state.selectedObject.id };
        }
        draw();
        return;
      }
    }
    if (state.selectedObject && state.selectedObject.type === 'apronLink' && !state.apronLinkDrawing) {
      const ah = hitTestApronLinkVertex(wx, wy);
      if (ah && ah.linkId === state.selectedObject.id) {
        pushUndo();
        state.dragApronLinkVertex = ah;
        state.selectedVertex = ah.kind === 'mid'
          ? { type: 'apronLink', id: ah.linkId, kind: 'mid', midIndex: ah.midIndex }
          : { type: 'apronLink', id: ah.linkId, kind: 'taxiway' };
        draw();
        return;
      }
    }
    state.selectedVertex = null;
    if ((mode === 'pbb' && state.pbbDrawing) || (mode === 'remote' && state.remoteDrawing) || (mode === 'holdingPoint' && state.holdingPointDrawing)) return;
    state.dragStart = { sx, sy, panX: state.panX, panY: state.panY };
    state.isPanning = false;
  });
  container.addEventListener('mousemove', function(ev) {
    const rect = canvas.getBoundingClientRect();
    const sx = ev.clientX - rect.left, sy = ev.clientY - rect.top;
    const [wx, wy] = screenToWorld(sx, sy);
    const snappedPt = worldPointToCellPoint(wx, wy, !!ev.shiftKey);
    const snappedPx = cellToPixel(snappedPt.col, snappedPt.row);
    const [col, row] = pixelToCell(wx, wy);
    if (coordEl) coordEl.textContent = 'cell: (' + col + ', ' + row + ')';
    const prev = state.hoverCell;
    state.hoverCell = { col, row };
    const hoverChanged = !prev || prev.col !== col || prev.row !== row;
    let drewThisMove = false;
    if (settingModeSelect.value === 'apronTaxiway' && state.apronLinkDrawing && state.apronLinkTemp) {
      const pw = state.apronLinkPointerWorld;
      if (!pw || pw[0] !== wx || pw[1] !== wy) {
        state.apronLinkPointerWorld = [wx, wy];
        scheduleDraw(); drewThisMove = true;
      }
    } else if (state.apronLinkPointerWorld) {
      state.apronLinkPointerWorld = null;
      scheduleDraw(); drewThisMove = true;
    }
    const pathLayoutDrawing = !!(state.terminalDrawingId || state.taxiwayDrawingId);
    const blockLayoutPathPtr = !!(state.isPanning || state.dragVertex || state.dragTaxiwayVertex || state.dragPbbBridgeVertex || state.dragStandConnection || state.dragApronLinkVertex || state.dragStandRotation);
    if (pathLayoutDrawing && !blockLayoutPathPtr) {
      const nx = snappedPx[0], ny = snappedPx[1];
      const lp = state.layoutPathDrawPointer;
      if (!lp || lp[0] !== nx || lp[1] !== ny) {
        state.layoutPathDrawPointer = [nx, ny];
        scheduleDraw(); drewThisMove = true;
      }
    } else if (state.layoutPathDrawPointer && (!pathLayoutDrawing || blockLayoutPathPtr)) {
      state.layoutPathDrawPointer = null;
      if (!drewThisMove) { scheduleDraw(); drewThisMove = true; }
    }
    if (state.dragVertex) {
      const term = state.terminals.find(t => t.id === state.dragVertex.terminalId);
      if (term && term.vertices[state.dragVertex.index]) {
        const v = term.vertices[state.dragVertex.index];
        v.col = snappedPt.col;
        v.row = snappedPt.row;
        scheduleDraw(); drewThisMove = true;
      }
      return;
    }
    if (state.dragTaxiwayVertex) {
      const tw = state.taxiways.find(t => t.id === state.dragTaxiwayVertex.taxiwayId);
      if (tw && tw.vertices[state.dragTaxiwayVertex.index]) {
        const v = tw.vertices[state.dragTaxiwayVertex.index];
        v.col = snappedPt.col;
        v.row = snappedPt.row;
        scheduleDraw(); drewThisMove = true;
        if (scene3d) update3DScene();
      }
      return;
    }
    if (state.dragStandRotation) {
      if (state.dragStandRotation.type === 'pbb') {
        const pbb = state.pbbStands.find(function(item) { return item.id === state.dragStandRotation.id; });
        if (pbb) {
          const origin = getPbbRotationOriginPx(pbb);
          const nextDeg = normalizeAngleDeg(Math.atan2(wy - origin[1], wx - origin[0]) * 180 / Math.PI);
          setPbbGeometryFromAngleLength(pbb, nextDeg, getPbbLengthMeters(pbb), true);
          const angleInput = document.getElementById('standAngle');
          if (angleInput) angleInput.value = String(Math.round(getPbbAngleDeg(pbb)));
          scheduleDraw(); drewThisMove = true;
          if (scene3d) update3DScene();
        }
      } else if (state.dragStandRotation.type === 'remote') {
        const st = state.remoteStands.find(function(item) { return item.id === state.dragStandRotation.id; });
        if (st) {
          const center = getRemoteStandCenterPx(st);
          const nextDeg = normalizeAngleDeg(Math.atan2(wy - center[1], wx - center[0]) * 180 / Math.PI);
          st.angleDeg = nextDeg;
          const angleInput = document.getElementById('remoteAngle');
          if (angleInput) angleInput.value = String(Math.round(nextDeg));
          scheduleDraw(); drewThisMove = true;
          if (scene3d) update3DScene();
        }
      }
      return;
    }
    if (state.dragPbbBridgeVertex) {
      const pbb = state.pbbStands.find(function(item) { return item.id === state.dragPbbBridgeVertex.pbbId; });
      if (pbb && Array.isArray(pbb.pbbBridges) && pbb.pbbBridges[state.dragPbbBridgeVertex.bridgeIndex] && Array.isArray(pbb.pbbBridges[state.dragPbbBridgeVertex.bridgeIndex].points) && pbb.pbbBridges[state.dragPbbBridgeVertex.bridgeIndex].points[state.dragPbbBridgeVertex.pointIndex]) {
        const pt = pbb.pbbBridges[state.dragPbbBridgeVertex.bridgeIndex].points[state.dragPbbBridgeVertex.pointIndex];
        if (state.dragPbbBridgeVertex.pointIndex === 0) {
          const projected = getClosestTerminalEdgePoint(wx, wy);
          if (projected && projected.point) {
            pt.x = projected.point[0];
            pt.y = projected.point[1];
          }
        } else {
          pt.x = snappedPx[0];
          pt.y = snappedPx[1];
        }
        scheduleDraw(); drewThisMove = true;
        if (scene3d) update3DScene();
      }
      return;
    }
    if (state.dragStandConnection) {
      const pbb = state.pbbStands.find(function(item) { return item.id === state.dragStandConnection.pbbId; });
      if (pbb) {
        pbb.apronSiteX = snappedPx[0];
        pbb.apronSiteY = snappedPx[1];
        scheduleDraw(); drewThisMove = true;
        if (scene3d) update3DScene();
      }
      return;
    }
    if (state.dragApronLinkVertex) {
      const lk = state.apronLinks.find(l => l.id === state.dragApronLinkVertex.linkId);
      if (!lk) {
        state.dragApronLinkVertex = null;
      } else if (state.dragApronLinkVertex.kind === 'mid') {
        const mi = state.dragApronLinkVertex.midIndex;
        if (lk.midVertices && mi >= 0 && mi < lk.midVertices.length &&
            col >= 0 && row >= 0 && col <= GRID_COLS && row <= GRID_ROWS) {
          lk.midVertices[mi].col = snappedPt.col;
          lk.midVertices[mi].row = snappedPt.row;
          scheduleDraw(); drewThisMove = true;
          if (scene3d) update3DScene();
        }
      } else if (state.dragApronLinkVertex.kind === 'taxiway') {
        const snap = snapWorldPointToTaxiwayPolyline(wx, wy, lk.taxiwayId);
        if (snap) {
          lk.tx = snap[0];
          lk.ty = snap[1];
          scheduleDraw(); drewThisMove = true;
          if (scene3d) update3DScene();
        }
      }
      return;
    }
    if (state.dragStart) {
      const dx = sx - state.dragStart.sx, dy = sy - state.dragStart.sy;
      if (!state.isPanning && (Math.abs(dx) > DRAG_THRESH || Math.abs(dy) > DRAG_THRESH))
        state.isPanning = true;
      if (state.isPanning) {
        state.panX = state.dragStart.panX + dx;
        state.panY = state.dragStart.panY + dy;
        scheduleDraw(); drewThisMove = true;
      }
    }
    const mode = settingModeSelect.value;
    if (!state.isPanning && !state.dragVertex && mode === 'holdingPoint' && state.holdingPointDrawing) {
      const snap = snapHoldingPointOnAllowedTaxiways(wx, wy);
      if (snap) {
        state.previewHoldingPoint = { x: snap.x, y: snap.y, pathType: snap.pathType };
      } else {
        state.previewHoldingPoint = null;
      }
      scheduleDraw(); drewThisMove = true;
    } else if (!state.isPanning && !state.dragVertex && mode === 'remote' && state.remoteDrawing) {
      const category = document.getElementById('remoteCategory').value || 'C';
      const angleDeg = normalizeAngleDeg(document.getElementById('remoteAngle') ? document.getElementById('remoteAngle').value : 0);
      const candidate = { x: snappedPx[0], y: snappedPx[1], category, angleDeg };
      const candCorners = getRemoteStandCorners(candidate);
      let overlap = false;
      for (let i = 0; i < state.remoteStands.length; i++) {
        if (rotatedRectsOverlap(candCorners, getRemoteStandCorners(state.remoteStands[i]))) { overlap = true; break; }
      }
      if (!overlap) {
        for (let i = 0; i < state.pbbStands.length; i++) {
          if (rotatedRectsOverlap(candCorners, getPBBStandCorners(state.pbbStands[i]))) { overlap = true; break; }
        }
      }
      const maxX = GRID_COLS * CELL_SIZE, maxY = GRID_ROWS * CELL_SIZE;
      if (candidate.x < 0 || candidate.y < 0 || candidate.x > maxX || candidate.y > maxY) overlap = true;
      state.previewRemote = { x: candidate.x, y: candidate.y, overlap };
      scheduleDraw(); drewThisMove = true;
    } else if (!state.isPanning && !state.dragVertex && mode === 'pbb' && state.pbbDrawing) {
      let bestEdge = null, bestD2 = Infinity;
      state.terminals.forEach(t => {
        if (!t.closed || t.vertices.length < 2) return;
        for (let i = 0; i < t.vertices.length; i++) {
          const v1 = t.vertices[i], v2 = t.vertices[(i+1) % t.vertices.length];
          const p1 = cellToPixel(v1.col, v1.row), p2 = cellToPixel(v2.col, v2.row);
          const near = closestPointOnSegment(p1, p2, snappedPx);
          if (near) {
            const d2 = dist2(near, snappedPx);
            if (d2 < bestD2) { bestD2 = d2; bestEdge = { near, p1, p2 }; }
          }
        }
      });
      const maxD2 = (CELL_SIZE*1.0)**2;
      if (bestEdge && bestD2 < maxD2) {
        const nearPt = bestEdge.near;
        const ex = (nearPt && nearPt[0] != null) ? nearPt[0] : 0;
        const ey = (nearPt && nearPt[1] != null) ? nearPt[1] : 0;
        const [x1,y1]=bestEdge.p1, [x2,y2]=bestEdge.p2;
        let nx = -(y2-y1), ny = x2-x1;
        const len = Math.hypot(nx,ny) || 1; nx /= len; ny /= len;
        const toClickX = snappedPx[0] - ex, toClickY = snappedPx[1] - ey;
        if (nx * toClickX + ny * toClickY < 0) { nx *= -1; ny *= -1; }
        const category = document.getElementById('standCategory').value || 'C';
        const standSize = getStandSizeMeters(category);
        const minLen = standSize / 2 + 3;
        const lenMeters = Number(document.getElementById('pbbLength').value || 15);
        const lenPx = Math.max(isFinite(lenMeters) && lenMeters > 0 ? lenMeters : 15, minLen);
        const px2 = ex + nx * lenPx, py2 = ey + ny * lenPx;
        const preview = { x1: ex, y1: ey, x2: px2, y2: py2, category };
        const overlap = pbbStandOverlapsExisting(preview);
        state.previewPbb = { x1: ex, y1: ey, x2: px2, y2: py2, category: preview.category, overlap };
        scheduleDraw(); drewThisMove = true;
      } else {
        if (state.previewPbb) { state.previewPbb = null; scheduleDraw(); drewThisMove = true; }
      }
    } else {
      let clearedPreview = false;
      if (state.previewRemote) { state.previewRemote = null; clearedPreview = true; }
      if (state.previewPbb) { state.previewPbb = null; clearedPreview = true; }
      if (state.previewHoldingPoint) { state.previewHoldingPoint = null; clearedPreview = true; }
      if (clearedPreview) { scheduleDraw(); drewThisMove = true; }
    }
    if (flightTooltip && !state.isPanning) {
      let tipDone = false;
      if (state.hasSimulationResult && state.globalUpdateFresh) {
        let bestFlight = null;
        let bestD2 = (CELL_SIZE * FLIGHT_TOOLTIP_CF) ** 2;
        const tSec = state.simTimeSec;
        state.flights.forEach(f => {
          const pose = getFlightPoseAtTimeForDraw(f, tSec);
          if (!pose || f.reg == null || !String(f.reg).trim()) return;
          const dx = pose.x - wx;
          const dy = pose.y - wy;
          const d2 = dx * dx + dy * dy;
          if (d2 < bestD2) { bestD2 = d2; bestFlight = f; }
        });
        if (bestFlight && bestFlight.reg) {
          flightTooltip.style.display = 'block';
          flightTooltip.textContent = String(bestFlight.reg).trim();
          flightTooltip.style.left = (ev.clientX + 12) + 'px';
          flightTooltip.style.top = (ev.clientY + 12) + 'px';
          tipDone = true;
        }
      }
      if (!tipDone) {
        const hit = hitTest(wx, wy);
        if (hit && hit.obj) {
          const name = (hit.obj.name != null && String(hit.obj.name).trim()) ? String(hit.obj.name).trim() : (hit.type === 'terminal' ? 'Building' : hit.type === 'pbb' ? 'Contact Stand' : hit.type === 'remote' ? 'Remote Stand' : hit.type === 'holdingPoint' ? holdingPointKindDisplayLabel(hit.obj.hpKind) : hit.type === 'taxiway' ? (hit.obj.name || 'Path') : hit.type === 'apronLink' ? (hit.obj.name || 'Apron Taxiway') : hit.type);
          flightTooltip.style.display = 'block';
          flightTooltip.textContent = name;
          flightTooltip.style.left = (ev.clientX + 12) + 'px';
          flightTooltip.style.top = (ev.clientY + 12) + 'px';
        } else {
          flightTooltip.style.display = 'none';
        }
      }
    }
    if (hoverChanged && !drewThisMove) { scheduleDraw(); drewThisMove = true; }
  });
  container.addEventListener('mouseleave', function() {
    state.dragStart = null;
    state.isPanning = false;
    state.dragStandRotation = null;
    state.dragPbbBridgeVertex = null;
    state.dragStandConnection = null;
    state.hoverCell = null;
    state.previewPbb = null;
    state.previewRemote = null;
    state.previewHoldingPoint = null;
    state.apronLinkPointerWorld = null;
    flushDrawNow();
  });
  container.addEventListener('dblclick', function(ev) {
    if (ev.button !== 0) return;
    const rect = canvas.getBoundingClientRect();
    const sx = ev.clientX - rect.left, sy = ev.clientY - rect.top;
    const [wx, wy] = screenToWorld(sx, sy);
