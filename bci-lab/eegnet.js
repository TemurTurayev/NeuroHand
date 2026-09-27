/*
 * EEGNet forward pass in plain JavaScript.
 *
 * Mirrors src/models/eegnet.py in eval mode (dropout off, batch-norm uses
 * running statistics), so the browser reaches the same probabilities as
 * PyTorch. Weights come from src/visualization/bci_lab_export.py, flattened in
 * PyTorch's row-major order.
 *
 * Works as a browser global (window.EEGNetJS) and as a Node module (tests).
 */
(function (root) {
  'use strict';

  var BN_EPS = 1e-5;

  function elu(v) { return v > 0 ? v : Math.exp(v) - 1; }

  function batchNormParams(w, prefix, n) {
    var scale = new Float64Array(n), shift = new Float64Array(n);
    for (var i = 0; i < n; i++) {
      var s = w[prefix + '.weight'][i] / Math.sqrt(w[prefix + '.running_var'][i] + BN_EPS);
      scale[i] = s;
      shift[i] = w[prefix + '.bias'][i] - w[prefix + '.running_mean'][i] * s;
    }
    return { scale: scale, shift: shift };
  }

  function Model(weights, arch) {
    this.w = weights;
    this.F1 = arch.F1;
    this.D = arch.D;
    this.F2 = arch.F2;
    this.K = arch.kernel_length;
    this.T = arch.n_samples;
    this.C = weights['depthwise_conv.weight'].length / (this.F1 * this.D);
    this.nClasses = weights['fc.bias'].length;
    this.bn1 = batchNormParams(weights, 'batchnorm1', this.F1);
    this.bn2 = batchNormParams(weights, 'batchnorm2', this.F1 * this.D);
    this.bn3 = batchNormParams(weights, 'batchnorm3', this.F2);
  }

  /**
   * @param {Float64Array[]} x  C channels × T samples, already z-scored.
   * @returns {{logits:number[], proba:number[], features:Float64Array[], bins:number}}
   */
  Model.prototype.forward = function (x) {
    var w = this.w, F1 = this.F1, D = this.D, F2 = this.F2, K = this.K;
    var C = this.C, T = this.T, FD = F1 * D;
    var pad1 = Math.floor(K / 2);
    var T1 = T + 2 * pad1 - K + 1;            // conv1 output length (1001)
    var w1 = w['conv1.weight'], wd = w['depthwise_conv.weight'];

    // Block 1: temporal conv -> BN1 -> depthwise spatial conv, fused per filter
    var z = [];
    for (var o = 0; o < FD; o++) z.push(new Float64Array(T1));
    var y = new Float64Array(T1);
    for (var f = 0; f < F1; f++) {
      var a = this.bn1.scale[f], b = this.bn1.shift[f];
      for (var c = 0; c < C; c++) {
        var xc = x[c];
        for (var t = 0; t < T1; t++) {
          var acc = 0, start = t - pad1;
          var k0 = start < 0 ? -start : 0;
          var k1 = Math.min(K, T - start);
          for (var k = k0; k < k1; k++) acc += w1[f * K + k] * xc[start + k];
          y[t] = a * acc + b;
        }
        for (var d = 0; d < D; d++) {
          var oo = f * D + d, wdc = wd[oo * C + c], zo = z[oo];
          for (var t2 = 0; t2 < T1; t2++) zo[t2] += wdc * y[t2];
        }
      }
    }

    // BN2 -> ELU -> AvgPool(1,4)
    var P1 = Math.floor(T1 / 4);
    var p = [];
    for (var o2 = 0; o2 < FD; o2++) {
      var s2 = this.bn2.scale[o2], h2 = this.bn2.shift[o2];
      var row = new Float64Array(P1);
      for (var i = 0; i < P1; i++) {
        var sum = 0;
        for (var j = 0; j < 4; j++) sum += elu(s2 * z[o2][i * 4 + j] + h2);
        row[i] = sum / 4;
      }
      p.push(row);
    }

    // Block 2: depthwise temporal conv (k=16, pad=8) -> pointwise 1×1
    var ws = w['separable_conv.weight'], wp = w['pointwise_conv.weight'];
    var K2 = 16, pad2 = 8, T2 = P1 + 2 * pad2 - K2 + 1;   // 251
    var u = [];
    for (var o3 = 0; o3 < FD; o3++) {
      var ur = new Float64Array(T2), pr = p[o3];
      for (var t3 = 0; t3 < T2; t3++) {
        var acc2 = 0;
        for (var k2 = 0; k2 < K2; k2++) {
          var idx = t3 + k2 - pad2;
          if (idx >= 0 && idx < P1) acc2 += ws[o3 * K2 + k2] * pr[idx];
        }
        ur[t3] = acc2;
      }
      u.push(ur);
    }

    // Pointwise -> BN3 -> ELU -> AvgPool(1,8)
    var bins = Math.floor(T2 / 8);            // 31
    var features = [];
    for (var g = 0; g < F2; g++) {
      var s3 = this.bn3.scale[g], h3 = this.bn3.shift[g];
      var v = new Float64Array(T2);
      for (var o4 = 0; o4 < FD; o4++) {
        var wgo = wp[g * FD + o4], uo = u[o4];
        for (var t4 = 0; t4 < T2; t4++) v[t4] += wgo * uo[t4];
      }
      var fr = new Float64Array(bins);
      for (var bi = 0; bi < bins; bi++) {
        var s = 0;
        for (var j2 = 0; j2 < 8; j2++) s += elu(s3 * v[bi * 8 + j2] + h3);
        fr[bi] = s / 8;
      }
      features.push(fr);
    }

    // Classifier
    var W = w['fc.weight'], B = w['fc.bias'], n = F2 * bins;
    var logits = [];
    for (var cl = 0; cl < this.nClasses; cl++) {
      var l = B[cl];
      for (var g2 = 0; g2 < F2; g2++)
        for (var b2 = 0; b2 < bins; b2++) l += W[cl * n + g2 * bins + b2] * features[g2][b2];
      logits.push(l);
    }
    return { logits: logits, proba: softmax(logits), features: features, bins: bins };
  };

  /** Contribution of every time bin to every class logit: [class][bin]. */
  Model.prototype.evidence = function (features, bins) {
    var W = this.w['fc.weight'], n = this.F2 * bins, out = [];
    for (var cl = 0; cl < this.nClasses; cl++) {
      var row = new Float64Array(bins);
      for (var b = 0; b < bins; b++) {
        var s = 0;
        for (var g = 0; g < this.F2; g++) s += W[cl * n + g * bins + b] * features[g][b];
        row[b] = s;
      }
      out.push(row);
    }
    return out;
  };

  /** Magnitude response of temporal filter f at the given frequencies (Hz). */
  Model.prototype.frequencyResponse = function (f, freqs, fs) {
    var w1 = this.w['conv1.weight'], K = this.K, out = new Float64Array(freqs.length);
    for (var i = 0; i < freqs.length; i++) {
      var re = 0, im = 0, omega = 2 * Math.PI * freqs[i] / fs;
      for (var k = 0; k < K; k++) {
        re += w1[f * K + k] * Math.cos(omega * k);
        im -= w1[f * K + k] * Math.sin(omega * k);
      }
      out[i] = Math.sqrt(re * re + im * im);
    }
    return out;
  };

  /** Spatial weights (one per electrode) of depthwise filter o. */
  Model.prototype.spatialFilter = function (o) {
    var wd = this.w['depthwise_conv.weight'], C = this.C;
    return Array.prototype.slice.call(wd, o * C, (o + 1) * C);
  };

  function softmax(v) {
    var m = Math.max.apply(null, v), e = v.map(function (x) { return Math.exp(x - m); });
    var s = e.reduce(function (a, b) { return a + b; }, 0);
    return e.map(function (x) { return x / s; });
  }

  /** Per-channel z-score (population std), same as the Python preprocessing. */
  function standardize(channels) {
    return channels.map(function (ch) {
      var n = ch.length, mean = 0, v = 0, i;
      for (i = 0; i < n; i++) mean += ch[i];
      mean /= n;
      for (i = 0; i < n; i++) v += (ch[i] - mean) * (ch[i] - mean);
      var sd = Math.sqrt(v / n) || 1, out = new Float64Array(n);
      for (i = 0; i < n; i++) out[i] = (ch[i] - mean) / sd;
      return out;
    });
  }

  var api = { Model: Model, standardize: standardize, softmax: softmax };
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.EEGNetJS = api;
})(this);
