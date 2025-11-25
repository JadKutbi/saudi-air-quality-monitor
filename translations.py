"""
Translations Module - Arabic/English language support

Provides bilingual support for the Saudi Arabia Air Quality Monitoring System.
"""

TRANSLATIONS = {
    "en": {
        # App title and header
        "app_title": "Saudi Arabia Air Quality Monitor",
        "app_subtitle": "Real-time pollution monitoring using Sentinel-5P satellite data",
        "time_label": "Time",

        # Sidebar
        "control_panel": "Control Panel",
        "select_city": "Select City",
        "choose_city_help": "Choose the city to monitor",
        "refresh_settings": "Refresh Settings",
        "auto_refresh": "Auto-refresh data",
        "refresh_interval": "Refresh interval (hours)",
        "last_update": "Last Update",
        "never": "Never",
        "language": "Language",

        # Cities
        "Yanbu": "Yanbu",
        "Jubail": "Jubail",
        "Jazan": "Jazan",

        # Tabs
        "tab_overview": "Overview",
        "tab_aqi": "AQI Dashboard",
        "tab_map": "Map View",
        "tab_analysis": "Analysis",
        "tab_violations": "Violations",
        "tab_insights": "Insights",
        "tab_history": "History",

        # Overview tab
        "current_metrics": "Current Air Quality Metrics",
        "no_data": "No data available",
        "fetching_data": "Fetching satellite data...",
        "data_age": "Data Age",
        "today": "today",
        "days_ago": "days ago",

        # Gas names
        "NO2": "Nitrogen Dioxide",
        "SO2": "Sulfur Dioxide",
        "CO": "Carbon Monoxide",
        "HCHO": "Formaldehyde",
        "CH4": "Methane",

        # Metrics
        "mean": "Mean",
        "max": "Max",
        "min": "Min",
        "threshold": "Threshold",
        "exceeded_by": "Exceeded by",
        "within_limits": "Within safe limits",

        # Violations
        "violation_analysis": "Violation Analysis",
        "no_violations": "No violations detected - Air quality is within safe limits",
        "violation_detected": "VIOLATION DETECTED",
        "severity": "Severity",
        "critical": "Critical",
        "moderate": "Moderate",
        "normal": "Normal",
        "hotspot_location": "Hotspot Location",
        "wind_conditions": "Wind Conditions",
        "wind_from": "Wind from",
        "wind_speed": "Speed",
        "ai_analysis": "AI Source Analysis",
        "analyzing": "Analyzing pollution source...",
        "nearby_factories": "Nearby Industrial Facilities",
        "upwind": "UPWIND",
        "distance": "Distance",
        "confidence": "Confidence",
        "already_saved": "Already saved",
        "saving": "Saving violation record...",
        "saved": "Saved",
        "save_failed": "Save failed",

        # Map
        "pollution_heatmap": "Pollution Heatmap",
        "select_gas": "Select Gas to Display",
        "violation_marker": "VIOLATION",
        "map_layers": "Map Layers",
        "satellite_view": "Satellite View",
        "factories_layer": "Industrial Facilities",

        # History
        "historical_trends": "Historical Trend Analysis",
        "timeline": "Timeline",
        "by_gas": "By Gas",
        "by_severity": "By Severity",
        "violations_over_time": "Violations Over Time",
        "avg_violations_day": "Avg Violations/Day",
        "peak_day": "Peak Day",
        "monitoring_period": "Monitoring Period",
        "total_violations": "Total Violations",
        "most_common_severity": "Most Common Severity",
        "most_frequent_gas": "Most Frequent Gas",
        "records_since": "Records Since",
        "filter_by_gas": "Filter by Gas",
        "show_records": "Show records",
        "clear_all": "Clear All",
        "delete": "Delete",
        "view_heatmap": "View Heatmap",
        "download_map": "Download Map (HTML)",
        "no_records": "No violation records found",
        "storage_info": "Storage Information",
        "cloud_storage": "Google Cloud Firestore - Persistent cloud storage enabled!",
        "local_storage": "Local Storage - Records may be lost on app restart",

        # AQI
        "aqi_dashboard": "Air Quality Index (AQI) Dashboard",
        "air_quality_status": "Air Quality Status",
        "dominant_pollutant": "Dominant Pollutant",
        "health_advice": "Health Advice",
        "aqi_good": "Good",
        "aqi_moderate": "Moderate",
        "aqi_unhealthy_sensitive": "Unhealthy for Sensitive Groups",
        "aqi_unhealthy": "Unhealthy",
        "aqi_very_unhealthy": "Very Unhealthy",
        "aqi_hazardous": "Hazardous",

        # Data quality
        "data_quality": "Data Quality Indicators",
        "spatial_coverage": "Spatial Coverage",
        "temporal_accuracy": "Temporal Accuracy",
        "measurement_validity": "Measurement Validity",
        "wind_sync": "Wind Sync",

        # Diagnostics
        "connection_diagnostics": "Connection Diagnostics",
        "test_connection": "Test Earth Engine Connection",
        "testing": "Testing connection...",
        "connection_success": "Connection successful!",
        "connection_failed": "Connection failed",

        # Common
        "all": "All",
        "unknown": "Unknown",
        "loading": "Loading...",
        "error": "Error",
        "success": "Success",
        "warning": "Warning",
        "info": "Info",
        "days": "days",
        "hours": "hours",
        "minutes": "minutes",
        "retry": "Retry",
        "violations": "violations",
        "km": "km",

        # Additional UI elements
        "about": "About",
        "monitored_gases": "Monitored Gases",
        "data_source": "Data Source",
        "standards": "Standards",
        "system_time": "System Time",
        "refresh_now": "Refresh Now",
        "detailed_analysis": "Detailed Analysis",
        "intelligent_insights": "Intelligent Insights & Predictions",
        "violation_details": "Violation Details",
        "aqi_dashboard_header": "Air Quality Index Dashboard",
        "pollution_map": "Pollution Map",
        "data_validation_report": "Data Validation Report",
        "quick_summary": "Quick Summary",
        "individual_gas_analysis": "Individual Gas Analysis",
        "detailed_values_table": "Detailed Values Table",
        "pollution_trends": "Pollution Trends",
        "showing_violations": "Showing {count} violation(s)",
        "no_data_available": "No pollution data available. Please try again later.",
        "connection_successful": "Earth Engine connection successful!",
        "connection_failed": "Connection failed",
        "can_access_data": "Can access Sentinel-5P data!",
        "cannot_access_data": "Cannot access Sentinel-5P",
        "using_service_account": "Using service account",
        "no_service_account": "No service account configured - using default auth",
        "please_check": "Please check",
        "violation_detected_for": "Violation Detected",
        "value": "Value",
        "wind": "Wind",
        "wind_confidence": "Wind Confidence",
        "type": "Type",
        "emissions": "Emissions",
        "satellite_pass": "Satellite Pass",
        "wind_reading": "Wind Reading",
        "sync_quality": "Sync Quality",
        "no_wind_data": "No wind data",
        "no_sync_data": "No sync data",
        "of_threshold": "of threshold",
        "normal_status": "Normal",
        "warning_status": "Warning",
        "record_deleted": "Record deleted",
        "failed_to_delete": "Failed to delete record",
        "all_records_cleared": "All records cleared",
        "click_to_confirm": "Click again to confirm deletion",
        "no_violations_recorded": "No violations recorded yet. Violations are automatically saved when detected.",
        "tip_violations": "Go to the Violations tab to detect and auto-save any current violations.",
        "tip": "Tip",
    },

    "ar": {
        # App title and header
        "app_title": "مراقب جودة الهواء في المملكة العربية السعودية",
        "app_subtitle": "مراقبة التلوث في الوقت الفعلي باستخدام بيانات القمر الصناعي Sentinel-5P",
        "time_label": "الوقت",

        # Sidebar
        "control_panel": "لوحة التحكم",
        "select_city": "اختر المدينة",
        "choose_city_help": "اختر المدينة للمراقبة",
        "refresh_settings": "إعدادات التحديث",
        "auto_refresh": "تحديث تلقائي للبيانات",
        "refresh_interval": "فترة التحديث (ساعات)",
        "last_update": "آخر تحديث",
        "never": "أبداً",
        "language": "اللغة",

        # Cities
        "Yanbu": "ينبع",
        "Jubail": "الجبيل",
        "Jazan": "جازان",

        # Tabs
        "tab_overview": "نظرة عامة",
        "tab_aqi": "مؤشر جودة الهواء",
        "tab_map": "الخريطة",
        "tab_analysis": "التحليل",
        "tab_violations": "المخالفات",
        "tab_insights": "الرؤى",
        "tab_history": "السجل",

        # Overview tab
        "current_metrics": "مقاييس جودة الهواء الحالية",
        "no_data": "لا توجد بيانات متاحة",
        "fetching_data": "جاري جلب بيانات القمر الصناعي...",
        "data_age": "عمر البيانات",
        "today": "اليوم",
        "days_ago": "أيام مضت",

        # Gas names
        "NO2": "ثاني أكسيد النيتروجين",
        "SO2": "ثاني أكسيد الكبريت",
        "CO": "أول أكسيد الكربون",
        "HCHO": "الفورمالديهايد",
        "CH4": "الميثان",

        # Metrics
        "mean": "المتوسط",
        "max": "الأقصى",
        "min": "الأدنى",
        "threshold": "الحد المسموح",
        "exceeded_by": "تجاوز بنسبة",
        "within_limits": "ضمن الحدود الآمنة",

        # Violations
        "violation_analysis": "تحليل المخالفات",
        "no_violations": "لا توجد مخالفات - جودة الهواء ضمن الحدود الآمنة",
        "violation_detected": "تم اكتشاف مخالفة",
        "severity": "الشدة",
        "critical": "حرج",
        "moderate": "متوسط",
        "normal": "طبيعي",
        "hotspot_location": "موقع البؤرة",
        "wind_conditions": "ظروف الرياح",
        "wind_from": "الرياح من",
        "wind_speed": "السرعة",
        "ai_analysis": "تحليل الذكاء الاصطناعي للمصدر",
        "analyzing": "جاري تحليل مصدر التلوث...",
        "nearby_factories": "المنشآت الصناعية القريبة",
        "upwind": "مصدر الرياح",
        "distance": "المسافة",
        "confidence": "نسبة الثقة",
        "already_saved": "محفوظ مسبقاً",
        "saving": "جاري حفظ سجل المخالفة...",
        "saved": "تم الحفظ",
        "save_failed": "فشل الحفظ",

        # Map
        "pollution_heatmap": "خريطة التلوث الحرارية",
        "select_gas": "اختر الغاز للعرض",
        "violation_marker": "مخالفة",
        "map_layers": "طبقات الخريطة",
        "satellite_view": "عرض القمر الصناعي",
        "factories_layer": "المنشآت الصناعية",

        # History
        "historical_trends": "تحليل الاتجاهات التاريخية",
        "timeline": "الجدول الزمني",
        "by_gas": "حسب الغاز",
        "by_severity": "حسب الشدة",
        "violations_over_time": "المخالفات عبر الزمن",
        "avg_violations_day": "متوسط المخالفات/اليوم",
        "peak_day": "يوم الذروة",
        "monitoring_period": "فترة المراقبة",
        "total_violations": "إجمالي المخالفات",
        "most_common_severity": "الشدة الأكثر شيوعاً",
        "most_frequent_gas": "الغاز الأكثر تكراراً",
        "records_since": "السجلات منذ",
        "filter_by_gas": "تصفية حسب الغاز",
        "show_records": "عرض السجلات",
        "clear_all": "مسح الكل",
        "delete": "حذف",
        "view_heatmap": "عرض الخريطة الحرارية",
        "download_map": "تحميل الخريطة (HTML)",
        "no_records": "لا توجد سجلات مخالفات",
        "storage_info": "معلومات التخزين",
        "cloud_storage": "Google Cloud Firestore - التخزين السحابي الدائم مفعّل!",
        "local_storage": "التخزين المحلي - قد تُفقد السجلات عند إعادة تشغيل التطبيق",

        # AQI
        "aqi_dashboard": "لوحة مؤشر جودة الهواء (AQI)",
        "air_quality_status": "حالة جودة الهواء",
        "dominant_pollutant": "الملوث الرئيسي",
        "health_advice": "نصائح صحية",
        "aqi_good": "جيد",
        "aqi_moderate": "متوسط",
        "aqi_unhealthy_sensitive": "غير صحي للفئات الحساسة",
        "aqi_unhealthy": "غير صحي",
        "aqi_very_unhealthy": "غير صحي جداً",
        "aqi_hazardous": "خطر",

        # Data quality
        "data_quality": "مؤشرات جودة البيانات",
        "spatial_coverage": "التغطية المكانية",
        "temporal_accuracy": "الدقة الزمنية",
        "measurement_validity": "صحة القياس",
        "wind_sync": "مزامنة الرياح",

        # Diagnostics
        "connection_diagnostics": "تشخيص الاتصال",
        "test_connection": "اختبار اتصال Earth Engine",
        "testing": "جاري اختبار الاتصال...",
        "connection_success": "الاتصال ناجح!",
        "connection_failed": "فشل الاتصال",

        # Common
        "all": "الكل",
        "unknown": "غير معروف",
        "loading": "جاري التحميل...",
        "error": "خطأ",
        "success": "نجاح",
        "warning": "تحذير",
        "info": "معلومات",
        "days": "أيام",
        "hours": "ساعات",
        "minutes": "دقائق",
        "retry": "إعادة المحاولة",
        "violations": "مخالفات",
        "km": "كم",

        # Additional UI elements
        "about": "حول",
        "monitored_gases": "الغازات المراقبة",
        "data_source": "مصدر البيانات",
        "standards": "المعايير",
        "system_time": "وقت النظام",
        "refresh_now": "تحديث الآن",
        "detailed_analysis": "التحليل المفصل",
        "intelligent_insights": "رؤى وتنبؤات ذكية",
        "violation_details": "تفاصيل المخالفة",
        "aqi_dashboard_header": "لوحة مؤشر جودة الهواء",
        "pollution_map": "خريطة التلوث",
        "data_validation_report": "تقرير التحقق من البيانات",
        "quick_summary": "ملخص سريع",
        "individual_gas_analysis": "تحليل كل غاز",
        "detailed_values_table": "جدول القيم المفصلة",
        "pollution_trends": "اتجاهات التلوث",
        "showing_violations": "عرض {count} مخالفة",
        "no_data_available": "لا توجد بيانات تلوث متاحة. يرجى المحاولة لاحقاً.",
        "connection_successful": "اتصال Earth Engine ناجح!",
        "connection_failed": "فشل الاتصال",
        "can_access_data": "يمكن الوصول إلى بيانات Sentinel-5P!",
        "cannot_access_data": "لا يمكن الوصول إلى Sentinel-5P",
        "using_service_account": "استخدام حساب الخدمة",
        "no_service_account": "لا يوجد حساب خدمة مكوّن - استخدام المصادقة الافتراضية",
        "please_check": "يرجى التحقق من",
        "violation_detected_for": "تم اكتشاف مخالفة",
        "value": "القيمة",
        "wind": "الرياح",
        "wind_confidence": "ثقة بيانات الرياح",
        "type": "النوع",
        "emissions": "الانبعاثات",
        "satellite_pass": "مرور القمر الصناعي",
        "wind_reading": "قراءة الرياح",
        "sync_quality": "جودة المزامنة",
        "no_wind_data": "لا توجد بيانات رياح",
        "no_sync_data": "لا توجد بيانات مزامنة",
        "of_threshold": "من الحد المسموح",
        "normal_status": "طبيعي",
        "warning_status": "تحذير",
        "record_deleted": "تم حذف السجل",
        "failed_to_delete": "فشل حذف السجل",
        "all_records_cleared": "تم مسح جميع السجلات",
        "click_to_confirm": "انقر مرة أخرى للتأكيد",
        "no_violations_recorded": "لا توجد مخالفات مسجلة بعد. يتم حفظ المخالفات تلقائياً عند اكتشافها.",
        "tip_violations": "اذهب إلى تبويب المخالفات لاكتشاف وحفظ المخالفات الحالية تلقائياً.",
        "tip": "نصيحة",
    }
}


def get_text(key: str, lang: str = "en") -> str:
    """
    Get translated text for a given key.

    Args:
        key: Translation key
        lang: Language code ('en' or 'ar')

    Returns:
        Translated string, or key if not found
    """
    return TRANSLATIONS.get(lang, TRANSLATIONS["en"]).get(key, key)


def get_direction(lang: str = "en") -> str:
    """Get text direction for the language."""
    return "rtl" if lang == "ar" else "ltr"


def get_font_family(lang: str = "en") -> str:
    """Get appropriate font family for the language."""
    if lang == "ar":
        return "'Noto Sans Arabic', 'Segoe UI', Tahoma, sans-serif"
    return "'Segoe UI', Tahoma, sans-serif"
