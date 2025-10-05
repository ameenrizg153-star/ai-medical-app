# ---- Diagnostic Script ----

print("1. بدء تشغيل السكربت...")

try:
    import pandas as pd
    print("2. تم استيراد مكتبة pandas بنجاح.")
except ImportError as e:
    print(f"خطأ فادح: لم يتم العثور على مكتبة pandas. الرجاء تشغيل 'pip install pandas'. الخطأ: {e}")
    exit()

try:
    from PIL import Image
    print("3. تم استيراد مكتبة Pillow (PIL) بنجاح.")
except ImportError as e:
    print(f"خطأ فادح: لم يتم العثور على مكتبة Pillow. الرجاء تشغيل 'pip install Pillow'. الخطأ: {e}")
    exit()

try:
    import fitz  # PyMuPDF
    print("4. تم استيراد مكتبة PyMuPDF (fitz) بنجاح.")
except ImportError as e:
    print(f"خطأ فادح: لم يتم العثور على مكتبة PyMuPDF. الرجاء تشغيل 'pip install PyMuPDF'. الخطأ: {e}")
    exit()

try:
    import pytesseract
    print("5. تم استيراد مكتبة pytesseract بنجاح.")
    
    # --- اختبار Tesseract ---
    # هذا السطر هو الاختبار الحقيقي. سيحاول العثور على Tesseract.
    # إذا تجمد البرنامج هنا، فالمشكلة 100% في إعداد Tesseract.
    print("6. جاري محاولة العثور على إصدار Tesseract...")
    version = pytesseract.get_tesseract_version()
    print(f"7. تم العثور على Tesseract بنجاح! الإصدار: {version}")

except ImportError as e:
    print(f"خطأ فادح: لم يتم العثور على مكتبة pytesseract. الرجاء تشغيل 'pip install pytesseract'. الخطأ: {e}")
    exit()
except Exception as e:
    print("\n--- مشكلة كبيرة! ---")
    print("حدث خطأ أثناء محاولة الاتصال بـ Tesseract.")
    print("السبب الأكثر شيوعًا هو أن Tesseract غير مثبت أو أن مساره غير صحيح.")
    print("الرجاء مراجعة خطوات تثبيت Tesseract في ملف README.md.")
    print(f"رسالة الخطأ التفصيلية: {e}")
    exit()

print("\n✅ نجح الاختبار! جميع المكتبات وإعداد Tesseract يعمل بشكل صحيح.")
print("الآن يمكنك إعادة الكود الأصلي إلى ملف app.py والمتابعة.")

