import os
import tempfile
import base64
from flask import Flask, request, render_template, jsonify, send_file
from werkzeug.utils import secure_filename
import pandas as pd
from traffic_person_reid import get_system

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['EXPORT_FOLDER'] = 'exports'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['EXPORT_FOLDER'], exist_ok=True)

reid_sys = get_system()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    """单图检索"""
    try:
        if 'query_img' not in request.files:
            return jsonify({'error': '未上传图片'}), 400
        file = request.files['query_img']
        if file.filename == '':
            return jsonify({'error': '文件名为空'}), 400

        filename = secure_filename(file.filename)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(temp_path)

        gallery_path = request.form.get('gallery_path', 'gallery').strip()
        use_yolo = request.form.get('use_yolo', 'true').lower() == 'true'

        frame_name = os.path.splitext(filename)[0]
        boxes_data = []
        all_results = []

        if use_yolo:
            persons, boxes = reid_sys.detect_persons(temp_path)
            if persons:
                for idx, person_img in enumerate(persons):
                    x1, y1, x2, y2 = boxes[idx]
                    person_id = idx + 1
                    boxes_data.append([x1, y1, x2, y2, person_id])
                    matches = reid_sys.search_similar_person(
                        person_img, gallery_path=gallery_path,
                        is_person_crop=True, top_k=5,
                        exclude_frame=frame_name
                    )
                    for rank, (p, s) in enumerate(matches):
                        all_results.append({
                            'person_id': person_id,
                            'rank': rank+1,
                            'name': os.path.basename(p),
                            'path': p,
                            'score': s
                        })
            else:
                matches = reid_sys.search_similar_person(
                    temp_path, gallery_path=gallery_path,
                    is_person_crop=False,
                    exclude_frame=frame_name
                )
                for i, (p, s) in enumerate(matches):
                    all_results.append({
                        'person_id': 0,
                        'rank': i+1,
                        'name': os.path.basename(p),
                        'path': p,
                        'score': s
                    })
        else:
            matches = reid_sys.search_similar_person(
                temp_path, gallery_path=gallery_path,
                is_person_crop=False,
                exclude_frame=frame_name
            )
            for i, (p, s) in enumerate(matches):
                all_results.append({
                    'person_id': 0,
                    'rank': i+1,
                    'name': os.path.basename(p),
                    'path': p,
                    'score': s
                })

        return jsonify({
            'boxes': boxes_data,
            'results': all_results
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/camera_search', methods=['POST'])
def camera_search():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': '无图像数据'}), 400

        img_data = base64.b64decode(data['image'].split(',')[1])
        temp_path = tempfile.mktemp(suffix='.jpg')
        with open(temp_path, 'wb') as f:
            f.write(img_data)

        gallery_path = data.get('gallery_path', 'gallery').strip()
        use_yolo = data.get('use_yolo', True)

        frame_name = None
        boxes_data = []
        all_results = []

        if use_yolo:
            persons, boxes = reid_sys.detect_persons(temp_path)
            if persons:
                for idx, person_img in enumerate(persons):
                    x1, y1, x2, y2 = boxes[idx]
                    person_id = idx + 1
                    boxes_data.append([x1, y1, x2, y2, person_id])
                    matches = reid_sys.search_similar_person(
                        person_img, gallery_path=gallery_path,
                        is_person_crop=True, top_k=5,
                        exclude_frame=frame_name
                    )
                    for rank, (p, s) in enumerate(matches):
                        all_results.append({
                            'person_id': person_id,
                            'rank': rank+1,
                            'name': os.path.basename(p),
                            'path': p,
                            'score': s
                        })
            else:
                matches = reid_sys.search_similar_person(
                    temp_path, gallery_path=gallery_path,
                    is_person_crop=False,
                    exclude_frame=frame_name
                )
                for i, (p, s) in enumerate(matches):
                    all_results.append({
                        'person_id': 0,
                        'rank': i+1,
                        'name': os.path.basename(p),
                        'path': p,
                        'score': s
                    })
        else:
            matches = reid_sys.search_similar_person(
                temp_path, gallery_path=gallery_path,
                is_person_crop=False,
                exclude_frame=frame_name
            )
            for i, (p, s) in enumerate(matches):
                all_results.append({
                    'person_id': 0,
                    'rank': i+1,
                    'name': os.path.basename(p),
                    'path': p,
                    'score': s
                })

        os.remove(temp_path)
        return jsonify({
            'boxes': boxes_data,
            'results': all_results
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/batch_search', methods=['POST'])
def batch_search():
    try:
        gallery_path = request.form.get('gallery_path', 'gallery').strip()
        use_yolo = request.form.get('use_yolo', 'true').lower() == 'true'

        files = request.files.getlist('query_imgs')
        if not files:
            return jsonify({'error': '未上传任何文件'}), 400

        batch_results = []

        for file in files:
            if file.filename == '':
                continue
            filename = secure_filename(file.filename)
            temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(temp_path)

            frame_name = os.path.splitext(filename)[0]
            image_data = {
                'filename': filename,
                'persons': []
            }

            if use_yolo:
                persons, boxes = reid_sys.detect_persons(temp_path)
                if persons:
                    for idx, person_img in enumerate(persons):
                        x1, y1, x2, y2 = boxes[idx]
                        person_id = idx + 1
                        matches = reid_sys.search_similar_person(
                            person_img, gallery_path=gallery_path,
                            is_person_crop=True, top_k=5,
                            exclude_frame=frame_name
                        )
                        person_results = []
                        for rank, (p, s) in enumerate(matches):
                            person_results.append({
                                'rank': rank+1,
                                'name': os.path.basename(p),
                                'path': p,
                                'score': s
                            })
                        image_data['persons'].append({
                            'person_id': person_id,
                            'box': [x1, y1, x2, y2],
                            'results': person_results
                        })
                else:
                    matches = reid_sys.search_similar_person(
                        temp_path, gallery_path=gallery_path,
                        is_person_crop=False,
                        exclude_frame=frame_name
                    )
                    person_results = []
                    for i, (p, s) in enumerate(matches):
                        person_results.append({
                            'rank': i+1,
                            'name': os.path.basename(p),
                            'path': p,
                            'score': s
                        })
                    image_data['persons'].append({
                        'person_id': 0,
                        'box': [],
                        'results': person_results
                    })
            else:
                matches = reid_sys.search_similar_person(
                    temp_path, gallery_path=gallery_path,
                    is_person_crop=False,
                    exclude_frame=frame_name
                )
                person_results = []
                for i, (p, s) in enumerate(matches):
                    person_results.append({
                        'rank': i+1,
                        'name': os.path.basename(p),
                        'path': p,
                        'score': s
                    })
                image_data['persons'].append({
                    'person_id': 0,
                    'box': [],
                    'results': person_results
                })

            batch_results.append(image_data)

        return jsonify({
            'success': True,
            'batch_results': batch_results
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/export_current', methods=['POST'])
def export_current():
    """导出当前显示的所有结果（修复：检查数据有效性）"""
    try:
        data = request.get_json()
        results = data.get('results', [])
        if not results:
            return jsonify({'error': '无数据可导出'}), 400

        export_data = []
        for item in results:
            export_data.append({
                '查询图片': item.get('query_img', ''),
                '行人编号': item.get('person_id', 1),
                '排名': item.get('rank', 0),
                '匹配图片': item.get('name', ''),
                '相似度': item.get('score', 0)
            })

        if not export_data:
            return jsonify({'error': '导出数据为空'}), 400

        df = pd.DataFrame(export_data)
        excel_path = os.path.join(app.config['EXPORT_FOLDER'], 'search_results.xlsx')
        df.to_excel(excel_path, index=False)
        return send_file(excel_path, as_attachment=True, download_name='检索结果.xlsx')
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/image')
def get_image():
    path = request.args.get('path')
    if not path or not os.path.exists(path):
        return '', 404
    return send_file(path, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)