import os
import math
from pathlib import Path
from functools import lru_cache
import joblib
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory, send_file
from io import BytesIO

ROOT = Path(__file__).resolve().parent
app = Flask(__name__, static_folder=str(ROOT), static_url_path='')
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024
BUNDLE_PATH = ROOT / 'rockfall_v4_k9.joblib'
B = joblib.load(BUNDLE_PATH) if BUNDLE_PATH.is_file() else None

@app.get('/')
def index():
    return send_from_directory(ROOT, 'index.html')

@app.get('/health')
def health():
    return jsonify(status='ok' if B else 'model_missing', model='v4-k9', loaded=bool(B)), (200 if B else 503)

@lru_cache(maxsize=1)
def worksheet_score_labels():
    """Read labels from the original CDOT worksheet, preserving score keys."""
    path = ROOT / '2007 CDOTslope scores.xls'
    if not path.is_file():
        return {}
    try:
        header = pd.read_excel(path, sheet_name='Slope Scores', header=None, nrows=2, engine='xlrd')
        groups = header.iloc[0].ffill()
        labels = header.iloc[1]
        result = {}
        for i in range(27, min(54, len(labels))):
            label = str(labels.iloc[i]).replace('\n', ' ').strip()
            group = str(groups.iloc[i]).replace('\n', ' ').strip()
            if label and label.lower() != 'nan':
                result[f'sc{i}'] = label
                if group and group.lower() != 'nan':
                    result[f'sc{i}'] = f'{label} ({group})'
        return result
    except Exception:
        app.logger.exception('Could not read worksheet score labels')
        return {}

@app.get('/schema')
def schema():
    if B is None:
        return jsonify(error='Model not exported yet'), 503
    from notebook_source import GEO, ENV, DER, TRAF, ITEMS, CAT
    return jsonify(geometry=GEO, environmental=ENV, derived=DER, traffic=TRAF,
                   geological_items={k:B['rank'][k][:9] for k in ITEMS},
                   item_names={**B['item_names'], **worksheet_score_labels()}, categories=B['category_values'], raters=sorted(B['rater_traffic_rule']),
                   note='Only the geological scores for the selected rock type are required.')

@app.post('/report/pdf')
def pdf_report():
    try:
        from pdf_report import create_pdf
        payload=request.get_json(silent=True)
        if not isinstance(payload,dict):return jsonify(error='Expected JSON object'),400
        pdf=create_pdf(payload.get('assessments'))
        return send_file(BytesIO(pdf),mimetype='application/pdf',as_attachment=True,download_name='rockfall-hazard-assessment-report.pdf')
    except (ValueError,TypeError,KeyError) as exc:
        return jsonify(error=str(exc)),400
    except Exception:
        app.logger.exception('PDF report generation failed')
        return jsonify(error='PDF report generation failed'),500

@app.post('/predict')
def predict():
    if B is None:
        return jsonify(error='Train and upload rockfall_v4_k9.joblib first'), 503
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify(error='Send a JSON object'), 400
    try:
        from notebook_source import GEO, ENV, DER, TRAF, ITEMS, ALL_ITEMS, BASE_NUM, CAT
        rock = str(data['rock_type']).strip()
        slope = str(data['slope_type']).strip()
        region = str(data['region']).strip()
        mode = 'cr' if rock.startswith('Cryst') else 'sed' if rock.startswith('Sed') else 'blk' if rock.startswith('Block') else None
        if mode is None or slope not in B['category_values']['slope_type'] or region not in B['category_values']['region'] or rock not in B['category_values']['rock_type']:
            raise ValueError('Invalid rock type, slope type or region')
        required = GEO + ENV + DER + TRAF + B['rank'][mode][:9]
        vals = {}
        for key in required:
            raw = data.get(key)
            if isinstance(raw, bool) or raw is None or raw == '':
                raise ValueError(f'Missing numeric field: {key}')
            val = float(raw)
            if not math.isfinite(val) or val < 0:
                raise ValueError(f'{key} must be a nonnegative finite number')
            vals[key] = val
        if vals['slope_angle'] > 90 or vals['ditch_slope'] > 90:
            raise ValueError('Slope angles must be at most 90 degrees')
        # Score values follow the original worksheet. Validate against its scoring scale.
        for key in ENV + DER + TRAF + B['rank'][mode][:9]:
            if vals[key] not in (0,3,9,27,81,243):
                raise ValueError(f'{key} must be one of 0, 3, 9, 27, 81, 243')
        rater = str(data.get('rater','')).strip()
        if rater and rater not in B['rater_traffic_rule']:
            raise ValueError('Unknown rater; choose a rater from the training dataset')
        traffic_rule = float(B['rater_traffic_rule'].get(rater, False))
        known = sum(vals.get(f'sc{i}',0) for i in B['known_ids']) + traffic_rule * sum(vals.get(t,0) for t in TRAF)
        observed = {f'obs_{item}': vals.get(item,0) if item in B['rank'][mode][:9] else 0 for item in ALL_ITEMS}
        osum = sum(vals[item] for item in B['rank'][mode][:9])
        row = {k:vals.get(k,0) for k in BASE_NUM}
        row.update(rock_type=rock, slope_type=slope, region=region, traffic_rule=traffic_rule)
        row.update(observed)
        frame = pd.DataFrame([row])[B['features']]
        pred_total = float(known + osum + sum(float(m.predict(frame)[0]) for m in B['models'])/len(B['models']))
        lo, hi = B['thresholds'].get(rock+'|'+slope, B['global_thresholds'])
        cls = 'Low' if pred_total <= lo else 'Moderate' if pred_total <= hi else 'High'
        return jsonify(hazard_class=cls, predicted_total_score=round(pred_total,2),
                       low_cutoff=round(lo,2), high_cutoff=round(hi,2),
                       model='v4-k9', probability_available=False,
                       note='This k=9 notebook does not include a fitted probability calibrator. Classification is score-threshold based.')
    except (ValueError, TypeError, KeyError) as e:
        return jsonify(error=str(e)),400
    except Exception:
        app.logger.exception('Prediction failed')
        return jsonify(error='Prediction failed'),500

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=int(os.environ.get('PORT',5000)))
