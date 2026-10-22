from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "technical_report" / "TreeLineWarning技术栈方法与解决方案.docx"

BLUE = "1F4E78"
PALE_BLUE = "EAF2F8"
PALE_GRAY = "F5F7F9"
GRID = "D9D9D9"
BLACK = RGBColor(0, 0, 0)


def set_cell_fill(cell, color: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), color)


def set_cell_margins(cell, top=100, start=120, bottom=100, end=120) -> None:
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for tag, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{tag}"))
        if node is None:
            node = OxmlElement(f"w:{tag}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_cell_borders(cell, color: str = GRID, size: int = 4) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    borders = tc_pr.first_child_found_in("w:tcBorders")
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tc_pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        tag = f"w:{edge}"
        element = borders.find(qn(tag))
        if element is None:
            element = OxmlElement(tag)
            borders.append(element)
        element.set(qn("w:val"), "single")
        element.set(qn("w:sz"), str(size))
        element.set(qn("w:color"), color)


def set_run_font(run, chinese="宋体", latin="Times New Roman", size=10.5, bold=None) -> None:
    run.font.name = latin
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), chinese)
    run._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), latin)
    run._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), latin)
    run.font.size = Pt(size)
    run.font.color.rgb = BLACK
    if bold is not None:
        run.bold = bold


def set_paragraph_format(paragraph, *, first_line=True, before=0, after=6, line=1.5) -> None:
    fmt = paragraph.paragraph_format
    fmt.space_before = Pt(before)
    fmt.space_after = Pt(after)
    fmt.line_spacing = line
    if first_line:
        fmt.first_line_indent = Pt(21)


def add_body(doc: Document, text: str, *, bold_lead: str | None = None) -> None:
    p = doc.add_paragraph()
    set_paragraph_format(p)
    if bold_lead and text.startswith(bold_lead):
        r1 = p.add_run(bold_lead)
        set_run_font(r1, bold=True)
        r2 = p.add_run(text[len(bold_lead):])
        set_run_font(r2)
    else:
        r = p.add_run(text)
        set_run_font(r)


def add_bullet(doc: Document, text: str) -> None:
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.left_indent = Pt(21)
    p.paragraph_format.first_line_indent = Pt(-10.5)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.35
    for r in p.runs:
        set_run_font(r)
    if not p.runs:
        set_run_font(p.add_run(text))
    else:
        p.runs[0].text = text


def add_heading(doc: Document, text: str, level: int) -> None:
    p = doc.add_paragraph(style=f"Heading {level}")
    p.paragraph_format.space_before = Pt(12 if level == 1 else 8)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(text)
    set_run_font(r, chinese="黑体", latin="Times New Roman", size=16 if level == 1 else 13, bold=True)


def add_table(doc: Document, headers: list[str], rows: list[list[str]], widths: list[float] | None = None) -> None:
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    table.rows[0]._tr.get_or_add_trPr().append(OxmlElement("w:tblHeader"))
    for idx, text in enumerate(headers):
        cell = table.rows[0].cells[idx]
        cell.text = text
        set_cell_fill(cell, BLUE)
        set_cell_margins(cell)
        set_cell_borders(cell)
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(0)
        for r in p.runs:
            set_run_font(r, chinese="黑体", size=9.5, bold=True)
            r.font.color.rgb = RGBColor(255, 255, 255)
    for r_idx, values in enumerate(rows):
        cells = table.add_row().cells
        for c_idx, text in enumerate(values):
            cell = cells[c_idx]
            cell.text = str(text)
            set_cell_fill(cell, "FFFFFF" if r_idx % 2 == 0 else PALE_BLUE)
            set_cell_margins(cell)
            set_cell_borders(cell)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if c_idx == 0 else WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.2
            for run in p.runs:
                set_run_font(run, size=9.2)
    if widths:
        for row in table.rows:
            for cell, width in zip(row.cells, widths, strict=True):
                cell.width = Cm(width)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def add_equation(doc: Document, text: str, explanation: str) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(3)
    r = p.add_run(text)
    set_run_font(r, latin="Cambria Math", size=11)
    p2 = doc.add_paragraph()
    p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p2.paragraph_format.space_after = Pt(6)
    r2 = p2.add_run(explanation)
    set_run_font(r2, size=9.2)


def add_page_number(section) -> None:
    footer = section.footer
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(3)
    run = p.add_run("第 ")
    set_run_font(run, size=9)
    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = "PAGE"
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    run._r.append(fld_begin)
    run._r.append(instr)
    run._r.append(fld_end)
    run2 = p.add_run(" 页")
    set_run_font(run2, size=9)


def configure_styles(doc: Document) -> None:
    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    normal.font.size = Pt(10.5)
    for name, size in (("Title", 22), ("Heading 1", 16), ("Heading 2", 13)):
        style = doc.styles[name]
        style.font.name = "Times New Roman"
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "黑体")
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = BLACK


def build() -> Path:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = Document()
    configure_styles(doc)
    sec = doc.sections[0]
    sec.top_margin = Cm(2.5)
    sec.bottom_margin = Cm(2.2)
    sec.left_margin = Cm(2.7)
    sec.right_margin = Cm(2.5)
    add_page_number(sec)

    title = doc.add_paragraph(style="Title")
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_before = Pt(95)
    title.paragraph_format.space_after = Pt(20)
    tr = title.add_run("TreeLineWarning 树线智能巡检预警系统技术报告")
    set_run_font(tr, chinese="黑体", size=22, bold=True)
    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.paragraph_format.space_after = Pt(12)
    sr = subtitle.add_run("技术栈 核心方法与工程解决方案")
    set_run_font(sr, chinese="黑体", size=15, bold=True)
    meta = doc.add_paragraph()
    meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    meta.paragraph_format.space_before = Pt(110)
    meta.paragraph_format.space_after = Pt(4)
    set_run_font(meta.add_run("项目名称：配电线路树线隐患智能检测与预测系统"), size=11)
    meta2 = doc.add_paragraph()
    meta2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_run_font(meta2.add_run("报告版本：2026 年 10 月"), size=11)
    doc.add_page_break()

    add_heading(doc, "摘要", 1)
    add_body(doc, "本系统面向输配电线路走廊中的树线净空巡检、风险预测和工单生成任务，形成从 LAS 或 LAZ 原始点云导入、坐标与单位确认、杆塔及导线建模、树木点云提取与人工校核、树高胸径冠幅测量、三维最小净空计算，到短期和长期生长预测、风致触线情景分析、风险分级与工单输出的完整离线工作流。系统的核心设计原则是保留原始点索引、区分科学计算点集与显示抽样点集、所有关键结果均能回溯到原始点云，并允许人工修正算法无法可靠判定的树种、胸径和树木归属。")
    add_body(doc, "工程实现采用 Python 作为算法与服务主语言，FastAPI 或本地 HTTP 服务负责接口，PyWebView 和 Edge WebView2 承载桌面界面，Giro3D 与 Three.js 完成三维点云交互，NumPy 与 SciPy 完成数值计算和空间邻域搜索。系统在无网络环境下可完成除气象局在线接入外的全部核心操作，并通过项目目录、版本化 JSON 和 NPZ 证据文件保存处理结果。")
    add_body(doc, "关键词：点云处理；单木分割；胸径测量；树线净空；数字孪生；季节生长；风险预警")

    add_heading(doc, "1 系统目标与总体技术路线", 1)
    add_body(doc, "系统要解决的核心问题不是单纯显示点云，而是将大规模原始点云转换为可复核、可预测、可追溯的树线巡检结果。总体技术路线分为数据层、几何计算层、业务预测层和桌面交互层。数据层保存原始 LAS 或 LAZ、坐标参考、原始点索引和阶段性成果；几何计算层完成地面、杆塔、导线、树木及净空关系计算；业务预测层根据树种、树龄、季节与气象数据推算生长和风险变化；桌面交互层负责人工复核、参数设置、结果查看、导出与工单生成。")
    add_table(doc, ["阶段", "输入", "核心处理", "输出"], [
        ["数据接入", "LAS/LAZ、GIS、外部表格", "能力扫描、坐标系与单位确认、格式解析", "规范化项目数据"],
        ["点云处理", "原始全分辨率点云", "地面识别、杆塔选取、导线重建、树木提取", "带原始索引的对象点集"],
        ["人工复核", "算法分割结果", "合并、分开、区域套索删除、误分团块剔除", "确定的单木成员关系"],
        ["量测分析", "树木点集与导线曲线", "树高、胸径、冠幅、三维最小净空", "逐树结构与风险指标"],
        ["预测预警", "树种、树龄、气象、阈值", "季节生长、风致偏移、阈值穿越日期", "短期长期预测与工单"],
        ["成果管理", "项目与预测结果", "版本化保存、另存为、批量导入导出", "Excel/CSV/TSV/TXT/JSON/PDF"],
    ], [1.9, 3.0, 6.2, 4.5])

    add_heading(doc, "2 所用技术栈", 1)
    add_heading(doc, "2.1 桌面与服务端", 2)
    add_table(doc, ["技术", "用途", "选择理由"], [
        ["Python 3.10 及以上", "点云算法、预测、数据管理、桌面启动", "科学计算生态完善，便于离线封装"],
        ["FastAPI 和 Uvicorn", "预测接口、表格解析、训练与结果服务", "接口类型清晰，便于前后端解耦和自动测试"],
        ["本地 ThreadingHTTPServer", "点云复核工作流的本机服务", "仅监听 127.0.0.1，离线可用且部署简单"],
        ["PyWebView 和 Edge WebView2", "Windows 桌面窗口、文件选择与另存为", "复用网页交互能力，同时保留原生文件对话框"],
        ["PyInstaller", "生成可分发的 Windows 程序", "将解释器、依赖和静态资源一并打包"],
    ], [3.1, 5.4, 7.1])
    add_heading(doc, "2.2 点云与空间计算", 2)
    add_table(doc, ["技术", "用途", "关键作用"], [
        ["laspy 与 lazrs", "LAS/LAZ 读取和压缩解码", "保留坐标、颜色、分类、强度和原始点序号"],
        ["NumPy", "向量化数组计算", "承载百万级点坐标、标签和距离批处理"],
        ["SciPy cKDTree", "邻域查询与地面插值", "避免逐点全量比较，降低时间复杂度"],
        ["局部 PCA", "线性度、平面度、垂直度和主方向", "区分树干、树冠、导线和杂物的局部几何特征"],
        ["pyproj 与 pyshp", "坐标转换与 GIS 文件读取", "支持项目坐标、经纬度和矢量要素统一"],
        ["ONNX Runtime", "离线模型推理", "无需联网即可加载树种识别等模型"],
    ], [3.1, 5.4, 7.1])
    add_heading(doc, "2.3 可视化 数据与测试", 2)
    add_table(doc, ["技术", "用途", "说明"], [
        ["Giro3D 与 Three.js", "三维点云、导线、危险点和交互选取", "支持旋转、俯视、侧视、点拾取和对象着色"],
        ["HTML5 CSS3 JavaScript", "统一桌面界面", "实现表单、弹窗、分栏拖动、进度和返回操作"],
        ["Leaflet 与离线 GeoJSON", "项目位置和行政区地图", "离网仍可显示边界，联网时可接入气象信息"],
        ["Plotly", "生长与净空变化曲线", "用于时间轴与多指标折线展示"],
        ["openpyxl", "Excel 读写", "支持项目结果和批量预测表格交换"],
        ["pytest 与 Playwright", "算法、接口和界面验收", "覆盖回归测试、按钮行为和端到端流程"],
    ], [3.1, 5.4, 7.1])

    add_heading(doc, "3 点云处理与对象建模方法", 1)
    add_heading(doc, "3.1 原始数据保真与证据链", 2)
    add_body(doc, "点云导入后首先检查文件格式、点数、坐标维度、颜色、强度、分类、坐标参考和线性单位。系统为进入科学计算的每个点保留 original index，并在阶段成果中保存来源摘要和配置摘要。界面为了流畅可以抽样显示，但树高、胸径、冠幅和净空均使用全分辨率成员点计算，避免把显示降采样误当成量测数据。")
    add_body(doc, "该方法解决了三个常见问题：一是处理后点云无法回到原始证据；二是显示抽样导致量测值漂移；三是人工合并或分开后旧结果被错误复用。每次编辑后，系统根据当前成员关系重新计算结构指标和净空结果，并以版本或摘要判断缓存是否仍然有效。")

    add_heading(doc, "3.2 杆塔和导线重建", 2)
    add_body(doc, "杆塔点云依据高度、垂直度、线性度、占地范围、连续高度分带和地面接触证据进行候选筛选，并由用户确认两个端部杆塔。杆塔和已确认导线属于受保护基础设施对象，后续树木合并、团块删除和区域套索不得改变其成员。导线重建在档距方向上识别长、细、连续且高度稳定的点带，将同一根导线按顺序拟合为折线段。对被树冠遮挡但在方向、标高和前后连续性上满足条件的较长点带，允许作为第三根或第四根导线的候选证据，而不是仅保留最明显的两根线。")

    add_heading(doc, "3.3 树木提取与单木分割", 2)
    add_body(doc, "单木分割采用树干优先、树冠补充的组合策略。算法先利用局部 PCA、高度分带和竖向连续性寻找树干候选，再通过冠层高度模型和局部顶点确定树冠种子。树干与树冠之间使用距离、连续路径、密度变化、冠层轮廓和竞争关系建立归属分数；在三维切片和局部团块层面完成约束合并，最终生成确定、待复核或未分配点。该策略比单纯按树冠聚类更适合相邻树冠互相重叠的场景。")
    add_body(doc, "车辆、栏杆、立杆和其他杂物通过高度范围、局部主方向、垂直结构、冠层连接性以及与已保护杆塔导线的关系进行剔除。算法仍保留人工编辑：Ctrl 或指定多选键可选择多棵树进行合并；分开粘连树时根据树干种子重新分配上部点；区域套索删除对圈选区域内的普通树木点无类别限制，从而能够删除被误判成树木的非树点。")

    add_heading(doc, "4 树木结构参数计算方法", 1)
    add_heading(doc, "4.1 树高", 2)
    add_body(doc, "树高以当前单木对应的原始全分辨率成员点为依据。最低点和最高点都来自同一棵树的原始成员集合，不使用去地面后的最低点，也不使用显示抽样点。")
    add_equation(doc, "H = zmax(raw tree) - zmin(raw tree)", "式中 H 为树高，zmax 和 zmin 分别为该树原始成员点的最高与最低高程。")

    add_heading(doc, "4.2 胸径", 2)
    add_body(doc, "胸径截面严格锚定在该树原始点云最低点以上 1.30 m。系统在目标高度附近设置有限厚度带，先依据树干底部中心限制搜索半径，再从候选树干点的水平坐标分布计算稳健直径。当前实现采用 X、Y 两方向的 90% 分位与 10% 分位之差，并取两者平均，降低少量离群点对结果的影响。")
    add_equation(doc, "zDBH = zmin(raw tree) + 1.30 m", "胸径截面基准与点云预处理后的最低点无关。")
    add_equation(doc, "DBH = 0.5 × [(Q90x - Q10x) + (Q90y - Q10y)]", "仅使用截面带内且靠近树干中心的原始树干点。")
    add_body(doc, "系统同时输出截面点数和质量等级。点数充分且直径在合理范围内时标记为已测；点数较少时标记为估计或需复核；完全没有可靠截面证据时显示 No 或未获取，不填假值，并允许用户录入现场测量值。这样既满足自动化，也避免不可靠结果进入预测。")

    add_heading(doc, "4.3 冠幅", 2)
    add_body(doc, "冠幅使用单木原始成员点在水平 XY 平面的几何分布计算，不使用三维包围盒对角线。先对 XY 坐标进行主成分分析，取得树冠主轴和与其垂直的次轴，再分别计算投影坐标的 99% 分位与 1% 分位之差。两轴中的较大值为长冠幅，较小值为短冠幅，平均冠幅取二者均值，冠幅面积按椭圆面积计算。分位数方法能抑制少量孤立点造成的冠幅异常增大。")
    add_equation(doc, "Cmean = (Cmajor + Cminor) / 2", "Cmajor 与 Cminor 分别为树冠主轴和次轴方向的稳健宽度。")
    add_equation(doc, "Acrown = π × Cmajor × Cminor / 4", "将水平树冠轮廓近似为椭圆得到冠幅面积。")

    add_heading(doc, "5 树线净空与危险点计算方法", 1)
    add_heading(doc, "5.1 三维最小净空", 2)
    add_body(doc, "导线被离散为连续折线段，树木使用全分辨率原始成员点。系统计算每个树点到候选导线线段的三维点到线段距离，并保存最小距离对应的树点、导线点、导线编号和原始点索引。为控制内存，计算按点块流式执行；导线段较少时采用向量化直接计算，导线段较多时使用线段包围盒层次结构缩小候选范围。")
    add_equation(doc, "dmin = min distance(Ptree, segmentwire)", "对一棵树的所有原始成员点和有效导线段求最小三维欧氏距离。")
    add_body(doc, "除了三维最小净空，系统还分解水平距离和垂直距离，便于解释树冠从哪个方向接近导线。危险点不是任意着色点，而是距离达到用户风险阈值的原始树点；界面可显示该点到最近导线点的证据线。风险阈值变化后重新分类危险点，避免继续使用通用默认标准。")

    add_heading(doc, "5.2 风险分类", 2)
    add_body(doc, "几何证据与风险策略分离。已确认接触才判定为危险；未接触但净空小于或等于用户设置的预警阈值时判定为预警；净空高于阈值时判定为安全。若接触证据或导线质量不足，系统不输出虚假的安全结论，而是标记为需要复核。阈值、单位、策略版本和计算时间均随结果保存。")

    add_heading(doc, "6 生长预测与数字孪生方法", 1)
    add_heading(doc, "6.1 树种和季节生长", 2)
    add_body(doc, "预测以每棵树为独立对象保存树种、树龄、树高、胸径、冠幅、观测日期和位置。树种既可从模型给出建议，也可由用户选择或输入；树龄可以人工填写，也可根据树高、胸径和冠幅形成经验估计，再由用户确认。不同树木不共享同一树种和树龄。")
    add_body(doc, "年度生长量不再平均分配到十二个月，而是使用物候权重。春夏分配较高权重，秋季逐步降低，冬季接近休眠；针叶、环孔材和常绿阔叶树使用不同的月度权重曲线。温度、日照、降雨、湿度、树龄和当前树高作为非线性特征输入树种生长模型。为避免模型外推过大，单月树高和胸径增量设置合理上限。")
    add_equation(doc, "ΔHmonth = Growthannual(species) × Wphenology(month, species) × Fweather", "月度增量由树种年生长参考、物候权重与气象修正共同决定。")

    add_heading(doc, "6.2 短时预测与长期预测", 2)
    add_body(doc, "短时预测支持 1 天、3 天、7 天、15 天或用户指定天数。每一天可分别输入风力等级、降雨量、湿度、温度和日照时长；也可联网从气象数据源读取。长期预测以月为时间步输入月平均气象，适合判断未来数月或数年内何时进入预警阈值。外部表格批量入口支持 Excel、CSV、TSV、TXT 和 JSON，解析后先预览，缺失或错误字段可人工修改再计算。")

    add_heading(doc, "6.3 点云数字孪生", 2)
    add_body(doc, "点云数字孪生以已经确定的处理后树木成员点为几何基准，不要求必须完成正式复核发布。系统预先计算时间序列参数，拖动时间轴时在浏览器内更新树高、冠幅、胸径和净空，而不是每次重新启动后台分析。树木变化采用分枝级局部伸长和树冠尺度变化，避免把整棵点云简单等比例放大。孪生视图下方同步显示树高、胸径、冠幅和有效净空折线，并绘制用户标准线；当净空越过阈值时，曲线和状态颜色随之变化。")

    add_heading(doc, "7 风致触线情景方法", 1)
    add_body(doc, "风致触线采用工程筛查模型，不把树木简单视为刚体。模型根据树种组给出绿色木材弹性模量范围，用胸径估计代表枝条直径，用冠幅和树高约束代表枝条长度及迎风面积，再依据悬臂梁挠度关系反算达到当前净空所需的阵风速度。输出结果同时给出中心估计、速度区间和蒲福风级范围。")
    add_equation(doc, "δ = F × L³ / (3EI)", "δ 为枝条端部挠度，F 为风荷载，L 为代表枝条长度，E 为弹性模量，I 为截面惯性矩。")
    add_equation(doc, "Fwind = 0.5 × ρ × Cd × A × V²", "根据空气密度、阻力系数、迎风面积和阵风速度计算等效风荷载。")
    add_body(doc, "该方法只用于巡检优先级筛查。枝条直径、含水率、根系约束、实际风向、断枝和导线摆动没有逐树实测时，不应宣称某一级风一定触线。系统因此同时显示假设范围、适用边界和需要现场复核的提示。")

    add_heading(doc, "8 主要问题与解决办法", 1)
    add_table(doc, ["问题", "原因", "解决办法", "效果"], [
        ["去噪后树干缺失", "仅按地面以上高度或冠层特征筛选", "保留原始索引；树干优先提取；杆塔导线设为保护对象；去噪只删除无关点", "树高和胸径仍可追溯到原始树点"],
        ["胸径高度不标准", "以处理后最低点或地面去除后的点为基准", "改用该树原始成员最低点加 1.30 m；无证据时显示 No 并允许人工输入", "消除预处理造成的截面上移"],
        ["冠幅测不出或异常", "用错误的三维范围或少量极端点", "对原始 XY 做 PCA，并采用 1% 到 99% 分位宽度", "每棵有效树都有可解释的长短冠幅"],
        ["相邻树粘连", "只按树冠聚类，缺乏树干约束", "先找树干种子，再向上关联树冠；提供多选合并和按树干重新分开", "改善并木、漏分和错分"],
        ["杂物被识别成树", "栏杆、立杆与树干局部几何相似", "结合高度、冠层连接、竖向连续性和基础设施保护；区域套索可强制删除普通点", "减少反复手工清理"],
        ["三四根导线只识别两根", "遮挡和分段导致连续证据不足", "按档距方向、标高、长度与前后连续性补齐候选导线，并保留质量标记", "提高导线拟合完整性"],
        ["五年后仍不触线或冬季增长过快", "按固定月增量或等比例缩放", "使用树种年增量与月度物候权重，春夏高、秋冬低，并逐日逐月积分", "预测曲线符合季节规律"],
        ["阈值修改后工单未同步", "计算结果和界面标准分离", "阈值进入版本化风险策略；危险点、预测、工单共用同一快照", "修改标准后可重新预测并保持一致"],
        ["项目重开后结果丢失", "结果仅保存在内存或临时页面", "每个项目建立独立目录，保存清单、阶段成果、模型快照和最新报告", "关闭软件后仍可继续项目"],
        ["离线环境无法使用", "依赖在线地图或云端模型", "本地 127.0.0.1 服务、离线 GeoJSON、离线模型和静态资源打包", "除在线气象外均可离网运行"],
        ["工单信息冗余", "逐树生成长页面且缺少档距层级", "按档距 1、档距 2 等分组，同一档距的所有树以精简表格集中展示", "巡检人员可快速定位风险树"],
    ], [2.8, 3.3, 6.7, 2.8])

    add_heading(doc, "9 数据安全与账户机制", 1)
    add_body(doc, "软件首次登录使用默认账号后，要求修改用户名和密码并完善姓名、手机号与头像。密码不以明文保存，而是使用随机盐和 PBKDF2 HMAC SHA 256 进行多轮派生后保存摘要。忘记密码时，系统核对用户名、姓名、规范化手机号和头像摘要，通过后允许设置新密码；找回流程只重置密码，不显示旧密码。账户资料与项目数据保存在本机用户目录，打开其他项目不会重置账户。")
    add_body(doc, "软件服务只绑定本机回环地址，外部网络设备不能直接访问。项目文件采用独立文件夹管理，导出和工单保存使用 Windows 原生另存为对话框，由用户选择磁盘和目录。关键配置、计算结果和报告记录版本号、时间和来源，便于审计与重复计算。")

    add_heading(doc, "10 测试验证方法", 1)
    add_body(doc, "测试分为算法单元测试、接口测试、浏览器自动化测试和真实点云流程测试。算法测试检查胸径是否使用原始最低点加 1.30 m、冠幅是否为水平稳健宽度、净空距离是否与精确参考算法一致、季节权重之和是否为 1。接口测试检查登录、找回密码、表格解析、预测、阈值修改、项目恢复和报告保存。浏览器测试使用 Playwright 实际点击按钮、切换页面、返回上一步、导入导出并捕获 JavaScript 错误。真实点云测试使用不同场景的 LAS 或 LAZ 从导入开始运行两至三遍，检查杆塔导线保护、树木编辑、结果复核、数字孪生和工单保存。")
    add_table(doc, ["验收对象", "重点检查", "通过标准"], [
        ["数据导入", "格式、点数、单位、坐标系、颜色和强度", "错误有明确提示，正确数据可继续"],
        ["点云处理", "杆塔导线不被删除，树干保留，杂物可清理", "对象成员与原始索引一致"],
        ["人工编辑", "合并、分开、套索、撤销和返回", "编辑后画面和结果同步更新"],
        ["结构量测", "树高、1.30 m 胸径、冠幅", "数值来源明确；无证据不造值"],
        ["风险预测", "季节、天气、风级、阈值和日期", "短期长期结果逻辑一致"],
        ["成果输出", "表格、工单、历史记录和另存为", "可选择路径且重开项目可恢复"],
    ], [2.9, 7.4, 5.3])

    add_heading(doc, "11 方法边界与后续改进", 1)
    add_body(doc, "点云自动识别受点密度、遮挡、扫描角度和分类质量影响。胸径截面没有足够树干回波时，应使用现场实测值；树龄估计属于经验模型，需要按地区和树种校准；风致触线属于情景筛查，不能替代结构力学试验和现场安全判断；在线气象数据依赖外部服务的可用性，离线时应允许人工输入或导入文件。")
    add_body(doc, "后续优化应优先积累带真实树种、树龄、胸径、枝条力学参数和多期点云的本地样本，建立分树种增量模型；其次应增加多期点云配准与真实生长变化验证；最后可将阈值规则、预测模型版本和工单处置结果形成闭环，用已完成巡检数据持续校准模型。")

    add_heading(doc, "12 结论", 1)
    add_body(doc, "TreeLineWarning 的技术路线以原始点云证据为基础，以三维几何计算为核心，以人工可修正为保障，再叠加季节生长、气象和风致偏移情景，实现从点云到预警工单的完整闭环。系统重点解决了树干在预处理后丢失、胸径基准错误、冠幅不稳定、相邻树粘连、导线遮挡、净空证据不可追溯、秋冬生长失真、阈值不同步、离线不可用和工单难阅读等问题。对于数据不足的指标，系统坚持显示未获取并允许人工补录，不以假值替代证据。该设计适合用于巡检辅助和风险排序，但最终处置仍应依据现行行业标准与现场复核。")

    # Keep all headings black and remove accidental direct theme color.
    for paragraph in doc.paragraphs:
        if paragraph.style.name in {"Title", "Heading 1", "Heading 2"}:
            for run in paragraph.runs:
                run.font.color.rgb = BLACK

    doc.save(OUT)
    return OUT


if __name__ == "__main__":
    print(build())
