"""Output contracts traced to prompt_config.py and the task scripts; no VLM imports."""
import math

SEATS = ['主驾', '副驾', '副驾后排', '后排中间', '主驾后排']
ACTIONS = ['头手伸出车窗外', '看手机', '看电脑或纸质资料', '打电话', '喝水', '吃东西',
           '化妆', '抽烟', '打哈欠', '打喷嚏/咳嗽', '擤鼻涕', '睡觉', '说话/唱歌',
           '跳舞', '大哭', '大笑', '看向窗外', '上车', '下车', '开车', '蹦跳', '倒立', '其他']
EVENTS = ['无风险', '开门杀', '行人破坏', '机动车/非机动接触', '起火', '洪水']
CABIN_EXACT = ['位置', '性别', '年龄', '情绪']
CABIN_TEXT = ['种族', '衣服颜色', '衣服种类', '衣服风格', '衣服材质', '发型', '配饰', '行为']
DETECTIONS = {
    'pd_detect': (['charging_pile', 'fuel_pump'], 2),
    'pd_detect_wo_No_limit': (['charging_pile', 'fuel_pump'], None),
    'env_detect': (['large_truck', 'tunnel'], 3),
    'env_detect_wo_No_limit': (['large_truck', 'tunnel'], None),
    'truck_wo_No_limit': (['large_truck'], None),
}
TASKS = list(DETECTIONS) + ['extreme_weather', 'ex_wea_fine_snow', 'is_exist_tunnel',
                          'guard_ai', 'parking_number', 'parking_environment', 'ims_action', 'rec_cabin']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fields_for(task):
    exact, text, sets, detections = [], [], [], []
    if task in DETECTIONS:
        detections = ['$']
    elif task in ('extreme_weather', 'ex_wea_fine_snow'):
        exact = ['target_type']
    elif task == 'is_exist_tunnel':
        exact = ['has_tunnel']
    elif task == 'guard_ai':
        exact, text = ['event'], ['description']
    elif task == 'parking_number':
        exact, text = ['parking_number', 'floor'], ['message']
    elif task == 'parking_environment':
        exact, text = ['floor'], ['environmental_description']
    elif task == 'ims_action':
        exact = [f'occupied.{s}' for s in SEATS]
        sets = [f'actions.{s}' for s in SEATS]
    elif task == 'rec_cabin':
        exact = [f'occupied.{s}' for s in SEATS]
        # Attributes are conditional; only reference-present fields are scored.
    return exact, text, sets, detections


def prepare_output(obj, task):
    """Validate raw output first; only normalize explicit prompt ambiguities."""
    if task in DETECTIONS:
        labels, limit = DETECTIONS[task]
        if isinstance(obj, dict):
            require(set(obj) == {'message', 'detections'} and obj['message'] == '未检测到相关物体'
                    and obj['detections'] == [], 'no-target response requires message and empty detections')
            return []
        require(isinstance(obj, list), 'detections must be an array')
        require(limit is None or len(obj) <= limit, f'too many detections (limit {limit})')
        for item in obj:
            require(isinstance(item, dict) and set(item) == {'target_type', 'bbox'}, 'detection requires target_type and bbox')
            require(item['target_type'] in labels, 'unknown detection category')
            box = item['bbox']
            require(isinstance(box, list) and len(box) == 4, 'bbox must contain four coordinates')
            require(all(type(x) in (int, float) and math.isfinite(x) for x in box), 'bbox coordinates must be finite numbers')
            require(box[2] > box[0] and box[3] > box[1], 'bbox must have positive area')
        return obj
    require(isinstance(obj, dict), 'task output must be an object')
    if task in ('extreme_weather', 'ex_wea_fine_snow'):
        labels = ['fog', 'rain', 'snow', 'uncertain', 'others']
        if task == 'ex_wea_fine_snow':
            labels.append('snow_covered')
        require(set(obj) == {'target_type'} and obj['target_type'] in labels, 'invalid weather output/category')
    elif task == 'is_exist_tunnel':
        require(set(obj) == {'has_tunnel'}, 'expected only has_tunnel')
        value = obj['has_tunnel']
        require(type(value) is bool or (isinstance(value, str) and value in ('true', 'false')), 'expected boolean or prompt example string true/false')
        return {'has_tunnel': value if type(value) is bool else value == 'true'}
    elif task in ('guard_ai', 'parking_number', 'parking_environment'):
        keys = {'guard_ai': ['event', 'description'], 'parking_number': ['parking_number', 'floor', 'message'],
                'parking_environment': ['floor', 'environmental_description']}[task]
        require(set(obj) == set(keys) and all(isinstance(obj[k], str) for k in keys), 'incorrect fields or non-string values')
        if task == 'guard_ai':
            require(obj['event'] in EVENTS, 'unknown sentinel event')
            require(bool(obj['description'].strip()) and len(obj['description']) <= 80, 'description must contain 1-80 characters')
        elif task == 'parking_environment':
            require(len(obj['environmental_description']) <= 50, 'environment description exceeds 50 characters')
        else:
            require(not obj['parking_number'] or obj['message'] == '', 'readable parking number requires empty message')
    elif task in ('ims_action', 'rec_cabin'):
        require(set(obj) <= set(SEATS), 'unknown seat name')
        occupied = {seat: seat in obj for seat in SEATS}
        if task == 'ims_action':
            for actions in obj.values():
                require(isinstance(actions, list) and 1 <= len(actions) <= 3, 'occupied seat requires 1-3 actions; omit empty seats')
                require(all(isinstance(x, str) and x in ACTIONS for x in actions), 'unknown action')
                require(len(actions) == len(set(actions)), 'duplicate action')
            return {'occupied': occupied, 'actions': {seat: sorted(obj.get(seat, [])) for seat in SEATS}}
        for seat, attrs in obj.items():
            require(isinstance(attrs, dict) and bool(attrs), 'occupied seat must have attribute object')
            require(set(attrs) <= set(CABIN_EXACT + CABIN_TEXT), 'unknown cabin attribute')
            require(all(isinstance(value, str) for value in attrs.values()), 'cabin attributes must be strings')
            if '位置' in attrs:
                require(attrs['位置'] == seat, 'attribute position conflicts with seat key')
        return {'occupied': occupied, 'attributes': obj}
    return obj


def partial_output(obj, task):
    """Normalize readable fields without requiring the entire response to conform."""
    if not isinstance(obj, dict):
        return obj
    if task in DETECTIONS and obj.get('detections') == []:
        return []
    if task == 'is_exist_tunnel':
        value = obj.get('has_tunnel')
        if isinstance(value, str) and value in ('true', 'false'):
            return {**obj, 'has_tunnel': value == 'true'}
    if task == 'ims_action':
        return {'occupied': {s: s in obj for s in SEATS},
                'actions': {s: obj.get(s, []) for s in SEATS}}
    if task == 'rec_cabin':
        return {'occupied': {s: s in obj for s in SEATS}, 'attributes': obj}
    return obj


def validate_field(raw, task, name, value):
    """Validate only this field. Root/key/count/cross-field constraints stay global."""
    if not task:
        return
    if task in DETECTIONS:
        # The target list is one atomic metric field: never silently discard bad boxes.
        prepare_output(value, task)
    elif task in ('extreme_weather', 'ex_wea_fine_snow', 'is_exist_tunnel'):
        prepare_output({name: value}, task)
    elif task == 'guard_ai':
        if name == 'event':
            require(isinstance(value, str) and value in EVENTS, 'unknown sentinel event')
        else:
            require(isinstance(value, str) and bool(value.strip()) and len(value) <= 80,
                    'description must contain 1-80 characters')
    elif task in ('parking_number', 'parking_environment'):
        require(isinstance(value, str), 'field must be a string')
        if name == 'environmental_description':
            require(len(value) <= 50, 'environment description exceeds 50 characters')
    elif name.startswith('occupied.'):
        require(isinstance(raw, dict) and type(value) is bool, 'unreadable seat occupancy')
    elif task == 'ims_action':
        seat = name.split('.')[1]
        if seat in raw:
            prepare_output({seat: raw[seat]}, task)
    elif task == 'rec_cabin':
        _, seat, attribute = name.split('.')
        require(isinstance(value, str), 'cabin attribute must be a string')
        if attribute == '位置':
            require(value == seat, 'attribute position conflicts with seat key')
