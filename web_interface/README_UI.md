# واجهة فحص الصور مع الموديل

هذه الواجهة أصبحت مربوطة مع موديل YOLO المدرب عن طريق السيرفر:

```text
model_web_app.py
```

السيرفر يفتح صفحة الواجهة، ويستقبل الصورة على endpoint اسمه:

```text
/predict
```

ثم يستخدم ملف:

```text
runs/detect/runs/yolo_polyp/polyp_combined/weights/best.pt
```

أو أي `best.pt` أحدث يجده تلقائيا داخل مجلد `runs`.

## التشغيل

من مجلد المشروع شغل:

```powershell
& C:\Users\TUF\AppData\Local\Programs\Python\Python313\python.exe .\model_web_app.py
```

ثم افتح:

```text
http://127.0.0.1:8765
```

إذا أردت تحديد الموديل يدويا:

```powershell
& C:\Users\TUF\AppData\Local\Programs\Python\Python313\python.exe .\model_web_app.py --weights .\runs\detect\runs\yolo_polyp\polyp_combined\weights\best.pt
```

## كيف تعمل

- تختار صورة منظار من الجهاز.
- الواجهة ترسل الصورة إلى `/predict`.
- السيرفر يشغل YOLO على الصورة.
- النتيجة ترجع بصيغة JSON فيها:
  - القرار: هل يوجد اشتباه أو لا.
  - confidence.
  - bounding boxes إذا وجد polyp.
- الواجهة ترسم المربعات فوق الصورة.

## تحليل مجلد كامل

يمكن أيضا اختيار مجلد كامل يحتوي صور منظار:

1. اضغط زر `Folder`.
2. اختر المجلد الذي يحتوي الصور.
3. اضغط `Analyze Folder`.
4. الواجهة سترسل الصور واحدة واحدة إلى الموديل.
5. ستظهر قائمة لكل الصور، وتوضح:
   - اسم الصورة.
   - هل يوجد polyp أو لا.
   - عدد detections.
   - إحداثيات كل bounding box بصيغة `x, y, w, h`.
6. يمكن ضغط زر `CSV` لتحميل ملف فيه نتائج كل الصور وإحداثيات الـ bounding boxes.

عند الضغط على أي نتيجة في القائمة، تظهر الصورة نفسها في منطقة العرض، وترسم الواجهة مكان الـ polyp فوق الصورة.

## شكل نتيجة API

```json
{
  "mode": "model",
  "verdict": "اشتباه polyp / tumor",
  "level": "danger",
  "confidence": 87,
  "detections": [
    {
      "label": "polyp",
      "confidence": 87,
      "box": {
        "x": 0.34,
        "y": 0.28,
        "width": 0.32,
        "height": 0.34
      }
    }
  ]
}
```

إحداثيات `box` تكون normalized بين `0` و `1`.

## ملاحظة

هذه الواجهة تساعد على تجربة الموديل، لكنها ليست تشخيصا طبيا نهائيا. القرار الطبي يحتاج مختص وبيانات تحقق أقوى.
