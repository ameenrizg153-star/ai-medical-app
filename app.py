import pandas as pd
import pytesseract
from PIL import Image
import fitz  # PyMuPDF
import re
import os

# --- إعداد Tesseract (قد تحتاج لتحديد المسار إذا لم يكن في PATH) ---
# إذا قمت بتثبيت Tesseract في مسار غير افتراضي، قم بإلغاء التعليق وتعديل السطر التالي
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe' # مثال لمسار ويندوز

class LabReportReader:
    def __init__(self, db_path):
        """
        تهيئة المحلل وقاعدة البيانات.
        """
        try:
            self.db = pd.read_csv(db_path)
            # تجهيز قاعدة البيانات للبحث
            self.db.fillna({'aliases': ''}, inplace=True)
            self.all_test_names = self._get_all_test_identifiers()
            print("تم تحميل قاعدة البيانات بنجاح.")
        except FileNotFoundError:
            print(f"خطأ: لم يتم العثور على ملف قاعدة البيانات في المسار: {db_path}")
            self.db = None

    def _get_all_test_identifiers(self):
        """
        يجمع كل الأسماء الممكنة للفحوصات (code, name_en, name_ar, aliases) لتسهيل البحث.
        """
        names = set(self.db['code'].str.lower())
        names.update(self.db['name_en'].str.lower())
        # قد نحتاج لمعالجة خاصة للأسماء العربية إذا كان OCR لا يدعمها جيدًا
        # names.update(self.db['name_ar'].str.lower()) 
        
        for alias_list in self.db['aliases']:
            if isinstance(alias_list, str):
                names.update([alias.strip().lower() for alias in alias_list.split(';')])
        return names

    def _extract_text_from_file(self, file_path):
        """
        يستخرج النص من ملف صورة أو PDF.
        """
        _, file_extension = os.path.splitext(file_path)
        text = ""
        try:
            if file_extension.lower() in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
                # استخدام Tesseract مع اللغة الإنجليزية والعربية (eng+ara)
                text = pytesseract.image_to_string(Image.open(file_path), lang='eng+ara')
            elif file_extension.lower() == '.pdf':
                doc = fitz.open(file_path)
                for page in doc:
                    text += page.get_text()
                doc.close()
            else:
                return None, f"صيغة الملف '{file_extension}' غير مدعومة."
            return text, None
        except Exception as e:
            return None, f"حدث خطأ أثناء قراءة الملف: {e}"

    def _find_tests_in_text(self, text):
        """
        يبحث في النص المستخرج عن أسماء فحوصات ونتائجها الرقمية.
        هذه دالة مبسطة وقد تحتاج لتحسينات كثيرة.
        """
        found_tests = {}
        lines = text.split('\n')
        
        # نمط Regex للبحث عن رقم (قد يكون عشريًا)
        # هذا النمط يحاول العثور على أرقام قد تكون نتائج
        value_pattern = re.compile(r'(\d+\.?\d*)')

        for i, line in enumerate(lines):
            line_lower = line.lower()
            for test_name in self.all_test_names:
                # البحث عن اسم الفحص في السطر
                if re.search(r'\b' + re.escape(test_name) + r'\b', line_lower):
                    # إذا وجدنا اسم الفحص، نبحث عن رقم في نفس السطر أو السطر التالي
                    match = value_pattern.search(line)
                    if match:
                        # التأكد من أن النتيجة ليست جزءًا من اسم الفحص نفسه (مثل vitamin d3)
                        if not match.group(1).isalpha():
                            # الحصول على الرمز الرسمي للفحص من قاعدة البيانات
                            test_code = self.get_test_code(test_name)
                            if test_code:
                                found_tests[test_code] = match.group(1)
                                break # ننتقل للسطر التالي بعد العثور على فحص
        return found_tests

    def get_test_code(self, query):
        """
        يحصل على الرمز الرسمي (code) للفحص من أي اسم أو alias.
        """
        query = query.lower().strip()
        # البحث بالرمز
        mask = self.db['code'].str.lower() == query
        if mask.any(): return self.db[mask].iloc[0]['code']
        # البحث بالاسم الإنجليزي
        mask = self.db['name_en'].str.lower() == query
        if mask.any(): return self.db[mask].iloc[0]['code']
        # البحث بالأسماء المستعارة
        mask = self.db['aliases'].str.contains(query, na=False, case=False)
        if mask.any(): return self.db[mask].iloc[0]['code']
        return None

    def analyze_report(self, file_path):
        """
        الوظيفة الكاملة: تقرأ الملف، تستخرج الفحوصات، وتحللها.
        """
        if self.db is None:
            print("قاعدة البيانات غير محملة.")
            return

        print(f"🔍 جاري قراءة الملف: {file_path}...")
        text, error = self._extract_text_from_file(file_path)
        if error:
            print(error)
            return

        print("✅ تم استخراج النص. جاري البحث عن الفحوصات...")
        found_tests = self._find_tests_in_text(text)

        if not found_tests:
            print("لم يتم العثور على أي فحوصات معروفة في الملف.")
            # print("\n--- النص المستخرج ---")
            # print(text)
            # print("--------------------")
            return

        print(f"\n--- 🔬 تم العثور على {len(found_tests)} فحصًا. إليك التحليل: ---\n")
        analyzer = LabTestAnalyzer('tests_database.csv') # استخدام المحلل القديم للتحليل النهائي
        for code, result in found_tests.items():
            analysis = analyzer.analyze_result(code, result, simple_format=True)
            print(analysis)

class LabTestAnalyzer:
    # هذا الكلاس هو نفسه من الرد السابق مع تعديل بسيط
    def __init__(self, db_path):
        self.db = pd.read_csv(db_path)
        self.db['low'] = pd.to_numeric(self.db['low'], errors='coerce')
        self.db['high'] = pd.to_numeric(self.db['high'], errors='coerce')
        self.db.fillna({'recommendation_low': 'لا توجد نصيحة محددة.', 'recommendation_high': 'لا توجد نصيحة محددة.'}, inplace=True)

    def analyze_result(self, test_code, result_value, simple_format=False):
        test_info = self.db[self.db['code'] == test_code].iloc[0]
        result_value = float(result_value)
        
        name_ar = test_info['name_ar']
        icon = test_info['icon']
        low, high = test_info['low'], test_info['high']
        
        status = "طبيعي"
        recommendation = ""

        if pd.notna(low) and result_value < low:
            status = "منخفض"
            recommendation = test_info['recommendation_low']
        elif pd.notna(high) and result_value > high:
            status = "مرتفع"
            recommendation = test_info['recommendation_high']

        if simple_format:
            # تنسيق بسيط ومباشر كما طلبت
            return f"{icon} {name_ar} — {result_value} — {status} → \"{recommendation if recommendation else 'ضمن المعدل الطبيعي.'}\""
        
        # ... يمكن ترك التنسيق المفصل هنا كخيار ...
        return f"تحليل {name_ar}: {status}. نصيحة: {recommendation}"


def main():
    """
    الدالة الرئيسية لتشغيل التطبيق.
    """
    reader = LabReportReader('tests_database.csv')
    if reader.db is None:
        return

    print("\nمرحباً بك في القارئ والمحلل الآلي لتقارير المختبر.")
    file_path = input("الرجاء إدخال المسار الكامل لملف الصورة أو PDF: ").strip()

    if not os.path.exists(file_path):
        print("خطأ: الملف غير موجود. الرجاء التأكد من المسار.")
        return
        
    reader.analyze_report(file_path)

if __name__ == "__main__":
    main()
