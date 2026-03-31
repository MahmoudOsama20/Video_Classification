import os
import traceback
from django.shortcuts import render
from django.core.files.storage import default_storage
from .ml.predictor import predict_video

def index(request):
    result = None
    error = None
    if request.method == 'POST' and request.FILES.get('video'):
        video = request.FILES['video']
        path = default_storage.save(f'uploads/{video.name}', video)
        full_path = default_storage.path(path)
        try:
            label, confidence = predict_video(full_path)
            result = {'label': label, 'confidence': confidence}
        except Exception as e:
            error = traceback.format_exc()
            print("PREDICTION ERROR:\n", error)
    return render(request, 'detector/index.html', {'result': result, 'error': error})