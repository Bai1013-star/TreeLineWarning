from __future__ import annotations

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "manual_output"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DOCX_PATH = OUT_DIR / "TreeLineWarning一体化软件2.0_全流程使用说明.docx"

SCREEN_OVERVIEW = Path(r"C:\Users\32637\AppData\Local\Temp\codex-clipboard-8b8c58b0-5fd5-4742-b8ea-4080abeecdae.png")
SCREEN_DATA = Path(r"C:\Users\32637\AppData\Local\Temp\codex-clipboard-c02f9ed6-c89a-4613-bf57-cb92330c0cda.png")


def font_path(name: str) -> str:
    candidates = {
        "song": [r"C:\Windows\Fonts\simsun.ttc", r"C:\Windows\Fonts\simsun.ttf"],
        "hei": [r"C:\Windows\Fonts\simhei.ttf", r"C:\Windows\Fonts\msyh.ttc"],
    }
    for candidate in candidates[name]:
        if Path(candidate).exists():
            return candidate
    return r"C:\Windows\Fonts\arial.ttf"


def make_diagrams() -> tuple[Path, Path, Path]:
    song = ImageFont.truetype(font_path("song"), 33)
    small = ImageFont.truetype(font_path("song"), 25)
    bold = ImageFont.truetype(font_path("hei"), 38)
    tiny = ImageFont.truetype(font_path("song"), 21)

    flow_path = OUT_DIR / "figure_workflow.png"
    im = Image.new("RGB", (1800, 650), "#0e1a23")
    d = ImageDraw.Draw(im)
    d.text((55, 35), "TreeLineWarning 一体化全流程", fill="#f2f7fa", font=bold)
    items = [
        ("建立项目", "选择独立项目文件夹"),
        ("导入数据", "LAS/LAZ + GIS/导线"),
        ("点云处理", "保留树干，去除无关物"),
        ("复核定稿", "拆分粘连树 + DBH复核"),
        ("树木预测", "选择1～多棵树"),
        ("孪生/报告", "时间拖动 + 工单导出"),
    ]
    x = 45
    for i, (title, sub) in enumerate(items):
        w = 245
        d.rounded_rectangle((x, 180, x + w, 410), radius=22, fill="#173346", outline="#41c7dd", width=4)
        tw = d.textbbox((0, 0), title, font=song)[2]
        d.text((x + (w-tw)/2, 225), title, fill="#ffffff", font=song)
        lines = sub.split(" + ")
        yy = 310
        for line in lines:
            sw = d.textbbox((0, 0), line, font=tiny)[2]
            d.text((x + (w-sw)/2, yy), line, fill="#b7d9e5", font=tiny)
            yy += 34
        if i < len(items)-1:
            d.line((x+w+8, 294, x+w+52, 294), fill="#42d392", width=8)
            d.polygon([(x+w+52, 281), (x+w+78, 294), (x+w+52, 307)], fill="#42d392")
        x += 290
    d.text((55, 500), "原则：原始点云保留 · 项目数据自动保存 · 标准实时联动 · 离线可运行", fill="#42d392", font=small)
    im.save(flow_path)

    dbh_path = OUT_DIR / "figure_dbh.png"
    im = Image.new("RGB", (1600, 820), "#f7fbfd")
    d = ImageDraw.Draw(im)
    d.text((70, 45), "胸径（DBH）必须以原始地面为基准", fill="#153447", font=bold)
    ground_y = 690
    d.line((100, ground_y, 1500, ground_y), fill="#4b7654", width=12)
    d.text((105, 710), "原始地面模型 / 原始 LAS 地面", fill="#30543a", font=small)
    d.rounded_rectangle((610, 155, 870, ground_y), radius=95, fill="#876647", outline="#563c2a", width=6)
    for yy in range(180, ground_y-30, 45):
        d.ellipse((690, yy, 710, yy+20), fill="#d1a46e")
    slice_y = ground_y - 350
    d.line((390, slice_y, 1090, slice_y), fill="#e34b4b", width=8)
    d.text((1110, slice_y-22), "原始地面 + 1.30 m 截面", fill="#b32121", font=song)
    d.line((520, ground_y, 520, slice_y), fill="#197a96", width=5)
    d.polygon([(507, slice_y+12), (520, slice_y-12), (533, slice_y+12)], fill="#197a96")
    d.polygon([(507, ground_y-12), (520, ground_y+12), (533, ground_y-12)], fill="#197a96")
    d.text((250, 490), "1.30 m", fill="#197a96", font=bold)
    d.ellipse((625, slice_y-42, 855, slice_y+42), outline="#e34b4b", width=8)
    d.text((65, 120), "树高：原始地面至树冠最高有效点", fill="#153447", font=small)
    d.text((65, 165), "胸径：原始 LAS 中 1.30 m 薄层的稳健圆/椭圆拟合", fill="#153447", font=small)
    d.text((65, 210), "预处理后的无地面点云只用于分割显示，不作为高度零点", fill="#c23b31", font=small)
    d.text((65, 760), "本版会恢复经垂直连续性验证的树干点；车辆、地面及噪声分类点不会被恢复。", fill="#153447", font=tiny)
    im.save(dbh_path)

    pred_path = OUT_DIR / "figure_prediction.png"
    im = Image.new("RGB", (1800, 770), "#0e1a23")
    d = ImageDraw.Draw(im)
    d.text((55, 30), "统一的 1～多棵树预测逻辑", fill="#ffffff", font=bold)
    boxes = [
        (80, 160, 470, 420, "选择数量 N", "N=1 即单木\nN>1 按顺序选编号"),
        (560, 160, 950, 420, "补充共同条件", "树种/树龄\n日期/经纬度/天气"),
        (1040, 160, 1430, 420, "执行预测", "季节生长 + 风偏\n采用当前自定义标准"),
        (1450, 160, 1750, 420, "输出", "逐树结果\n孪生/工单/导出"),
    ]
    for idx, (x1,y1,x2,y2,title,sub) in enumerate(boxes):
        d.rounded_rectangle((x1,y1,x2,y2), radius=24, fill="#173346", outline="#41c7dd", width=4)
        d.text((x1+25,y1+38), title, fill="#ffffff", font=song)
        yy=y1+120
        for line in sub.split("\n"):
            d.text((x1+25,yy), line, fill="#b7d9e5", font=small); yy+=46
        if idx < len(boxes)-1:
            d.line((x2+10, 290, boxes[idx+1][0]-22, 290), fill="#42d392", width=8)
            d.polygon([(boxes[idx+1][0]-22,278),(boxes[idx+1][0]-2,290),(boxes[idx+1][0]-22,302)], fill="#42d392")
    d.text((80, 510), "预测结果必须同时给出：当前/预测净空、到达阈值日期与天数、预计接触风级、风险等级。", fill="#f3d477", font=small)
    d.text((80, 570), "标准变更后旧报告失效；重新预测后，工单与预测区使用同一套最新标准。", fill="#f3d477", font=small)
    d.text((80, 640), "外部表格批量：表格只录树木与树线信息；日期、地点、天气在表格外统一设置。", fill="#b7d9e5", font=small)
    im.save(pred_path)
    return flow_path, dbh_path, pred_path


def set_cell_shading(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def set_repeat_table_header(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def set_run_font(run, chinese="宋体", latin="Times New Roman", size=10.5, bold=None, color=None):
    run.font.name = latin
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), chinese)
    run._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), latin)
    run._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), latin)
    run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if color:
        run.font.color.rgb = RGBColor(*color)


def add_text(p, text, *, bold=False, size=10.5, color=None):
    r = p.add_run(text)
    set_run_font(r, size=size, bold=bold, color=color)
    return r


def add_para(doc, text="", *, style=None, indent=True, keep=False):
    p = doc.add_paragraph(style=style)
    add_text(p, text)
    p.paragraph_format.line_spacing = 1.5
    p.paragraph_format.space_after = Pt(5)
    if indent:
        p.paragraph_format.first_line_indent = Cm(0.74)
    if keep:
        p.paragraph_format.keep_with_next = True
    return p


def add_bullet(doc, text, level=0):
    p = doc.add_paragraph(style="List Bullet" if level == 0 else "List Bullet 2")
    add_text(p, text)
    p.paragraph_format.line_spacing = 1.35
    p.paragraph_format.space_after = Pt(3)
    return p


def add_step(doc, no, title, body):
    p = doc.add_paragraph()
    add_text(p, f"{no}. {title}：", bold=True, size=11)
    add_text(p, body, size=10.5)
    p.paragraph_format.line_spacing = 1.5
    p.paragraph_format.space_after = Pt(5)
    p.paragraph_format.left_indent = Cm(0.25)
    return p


def add_note(doc, title, text, color="DCEEF5"):
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = table.cell(0, 0)
    set_cell_shading(cell, color)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
    p = cell.paragraphs[0]
    add_text(p, title + "：", bold=True, color=(20, 73, 95))
    add_text(p, text)
    p.paragraph_format.space_after = Pt(0)
    doc.add_paragraph().paragraph_format.space_after = Pt(1)


def add_table(doc, headers, rows, widths=None, font_size=9):
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr = table.rows[0]
    set_repeat_table_header(hdr)
    for i, h in enumerate(headers):
        cell = hdr.cells[i]
        set_cell_shading(cell, "173346")
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        add_text(p, h, bold=True, size=font_size, color=(255,255,255))
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
    for ridx, row in enumerate(rows):
        cells = table.add_row().cells
        for i, val in enumerate(row):
            if ridx % 2:
                set_cell_shading(cells[i], "EDF5F8")
            p = cells[i].paragraphs[0]
            add_text(p, str(val), size=font_size)
            p.paragraph_format.space_after = Pt(0)
            cells[i].vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
    if widths:
        for row in table.rows:
            for i, width in enumerate(widths):
                row.cells[i].width = Cm(width)
    doc.add_paragraph().paragraph_format.space_after = Pt(1)
    return table


def add_page_number(paragraph):
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run()
    fld_char1 = OxmlElement("w:fldChar")
    fld_char1.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = " PAGE "
    fld_char2 = OxmlElement("w:fldChar")
    fld_char2.set(qn("w:fldCharType"), "end")
    run._r.extend([fld_char1, instr, fld_char2])
    set_run_font(run, size=9)


def build_document():
    flow_path, dbh_path, pred_path = make_diagrams()
    doc = Document()
    sec = doc.sections[0]
    sec.top_margin = Cm(2.4); sec.bottom_margin = Cm(2.2)
    sec.left_margin = Cm(2.6); sec.right_margin = Cm(2.3)
    sec.header_distance = Cm(1.2); sec.footer_distance = Cm(1.2)

    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Times New Roman"; normal.font.size = Pt(10.5)
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    for name, size in [("Title", 26), ("Heading 1", 16), ("Heading 2", 13), ("Heading 3", 11)]:
        st = styles[name]
        st.font.name = "Times New Roman"; st.font.size = Pt(size); st.font.bold = True
        st.font.color.rgb = RGBColor(21, 52, 71)
        st._element.rPr.rFonts.set(qn("w:eastAsia"), "黑体")
        st.paragraph_format.space_before = Pt(10)
        st.paragraph_format.space_after = Pt(6)
        st.paragraph_format.keep_with_next = True

    header = sec.header.paragraphs[0]
    header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    add_text(header, "TreeLineWarning 一体化软件 2.0　全流程使用说明", size=9, color=(88,110,122))
    add_page_number(sec.footer.paragraphs[0])

    # Cover
    for _ in range(4): doc.add_paragraph()
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_text(p, "TreeLineWarning 一体化软件 2.0", bold=True, size=27, color=(21,52,71))
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_text(p, "全流程使用说明与功能手册", bold=True, size=22, color=(25,122,150))
    doc.add_paragraph()
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_text(p, "点云处理 · 导线重建 · 单木复核 · 树木预测 · 点云孪生 · 预警工单", size=13, color=(60,90,105))
    doc.add_paragraph()
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_text(p, "版本日期：2026 年 9 月 30 日", size=12)
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_text(p, "适用平台：Windows 10 / Windows 11（64 位）", size=11)
    for _ in range(4): doc.add_paragraph()
    add_note(doc, "重要", "软件所有项目数据默认保存在 D 盘一体化软件文件夹内。再次打开项目时，请从“项目数据”目录选择对应项目；不需要重新导入点云。", "E7F5EE")
    doc.add_page_break()

    doc.add_heading("1　版本说明与本次重点修正", level=1)
    add_para(doc, "本手册对应 TreeLineWarning 一体化软件 2.0（2026-09-30 修订版）。本版将点云处理、人工复核、数据查看、1～多棵树预测、点云孪生、报告工单和历史记录统一到同一个桌面软件中。")
    changes = [
        "去噪不再把已经通过垂直连续性验证的原始树干点删除；车辆、地面、孤立噪声等无关点仍会被排除。",
        "胸径以原始 LAS 地面为高度零点，在“原始地面 + 1.30 m”截面拟合，不以去地面后的点云最低点计算。",
        "人工复核新增“分开粘连树”，可把同一编号内相连的两棵树自动拆开，且拆分后的点互不重叠、不丢失。",
        "胸径复核不再要求先完成最终定稿；分析完成后即可选择树木并打开原始截面。",
        "查看数据支持刷新和系统“另存为”，可导出 Excel、CSV、TSV、TXT、JSON。",
        "预测入口统一为“树木预测（1～多棵）”：选择 1 棵就是单木预测，选择多棵则按顺序生成结果与工单。",
        "树种列表与模型训练一致，并支持直接输入未收录树种；经纬度可手动输入。",
        "标准设置增加“恢复默认”；修改标准后，预测、风险判断和工单使用同一套实时标准。",
        "预测结果直接显示预计接触日期/天数及预计接触风级。",
    ]
    for x in changes: add_bullet(doc, x)
    add_note(doc, "旧项目提示", "旧项目保存的是修改前的分割结果。为了让树干恢复和新的胸径算法真正写入结果，请打开旧项目后从“单木分割”重新运行，再进入人工复核。", "FFF2CC")
    doc.add_heading("1.1　软件位置", level=2)
    add_table(doc, ["内容", "位置"], [
        ["主程序", r"D:\TreeLineWarning_一体化软件_20260926\TreeLineWarning一体化软件2.0_20260926\TreeLineWarning一体化软件2.0.exe"],
        ["桌面式快捷入口", r"…\00_启动TreeLineWarning一体化软件2.0.lnk"],
        ["项目数据", r"…\项目数据\项目名称"],
        ["导出结果", r"…\导出结果（也可在另存为窗口自选位置）"],
    ], widths=[3.2, 12.5], font_size=8.5)
    doc.add_page_break()

    doc.add_heading("2　完整工作流程", level=1)
    doc.add_picture(str(flow_path), width=Inches(6.55))
    p = doc.paragraphs[-1]; p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap = doc.add_paragraph(); cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_text(cap, "图 2-1　一体化软件推荐操作顺序", size=9)
    for n, title, body in [
        (1,"新建或打开项目","首次使用点击“新建项目”；继续上次任务点击“打开项目”，选择“项目数据”中的项目文件夹。"),
        (2,"导入点云与线路","导入 LAS/LAZ；若有线路 GIS 文件则导入，否则在导线重建页从点云识别并人工确认。"),
        (3,"顺序处理","依次完成点云预处理、地面点去除、导线重建和单木分割。"),
        (4,"人工复核","检查树木编号；对粘连树使用“分开粘连树”，对误分块执行合并/删除，并完成胸径复核。"),
        (5,"查看数据","刷新表格，检查树高、胸径、冠幅和净空；需要时另存为 Excel/CSV 等。"),
        (6,"树木预测","选择 1～多棵树、树种/树龄、起止日期、坐标、天气来源和标准后执行预测。"),
        (7,"孪生与输出","拖动时间轴观察树冠、树高、净空变化；生成预测报告与工单并归档。"),
    ]: add_step(doc,n,title,body)
    add_note(doc, "推荐", "每完成一个处理阶段先检查底部绿色状态块；只有上一步已完成，下一步数据才可靠。", "E7F5EE")
    doc.add_page_break()

    doc.add_heading("3　主界面与项目管理", level=1)
    if SCREEN_OVERVIEW.exists():
        doc.add_picture(str(SCREEN_OVERVIEW), width=Inches(6.55))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER
        add_text(p,"图 3-1　主界面布局示例（按钮文字以当前修订版为准）",size=9)
    add_para(doc, "顶部工具栏始终保持不变，点击任一功能只替换中间工作区，不会把顶部任务栏一起换掉。左侧为项目和线路结构，中间为当前任务，右侧为结果摘要；分隔条可拖动以放大中间点云区域。")
    add_table(doc,["按钮","用途","注意事项"],[
        ["新建项目","建立新的独立工程目录","项目名应能区分线路、日期或区段"],
        ["打开项目","继续之前保存的工程","选择“项目数据”下对应项目文件夹"],
        ["导入 LAS/LAZ","导入原始点云","原文件会被项目引用/缓存，不能随意移动源文件"],
        ["导入 GIS","导入线路或地理数据","无 GIS 时可用导线重建"],
        ["历史记录","查看既往预测与工单","用于追溯，不替代当前项目文件"],
    ],widths=[3.0,5.0,7.7])
    doc.add_heading("3.1　项目保存和再次打开", level=2)
    add_para(doc, "软件每次处理、复核、标准修改和预测都会写入当前项目。项目文件夹包含输入引用、运行结果、复核版本、预测缓存和工单记录。退出软件后无需另行执行“保存工程”；下次使用“打开项目”即可继续。")
    add_note(doc,"不要删除", "项目数据目录、原始点云缓存和运行结果之间存在关联。请只通过软件打开项目，不要手工删除项目内部文件。", "FCE8E6")
    doc.add_page_break()

    doc.add_heading("4　点云处理与树干保留", level=1)
    add_table(doc,["阶段","功能","检查重点"],[
        ["点云预处理","读取坐标、颜色、分类并建立本地缓存","点数和坐标范围是否合理"],
        ["地面点去除","建立地面模型并标记非地面点","显示可去地面，但原始地面仍保存供树高/DBH使用"],
        ["导线重建","识别导线点并拟合跨档曲线","导线方向、杆塔端点和弧垂"],
        ["单木分割","提取植被、识别树干并形成单木成员","树干是否完整、粘连树是否需拆分"],
    ],widths=[2.6,6.0,7.2])
    doc.add_heading("4.1　去噪与树干保护原则", level=2)
    add_para(doc, "本版把“去噪”和“删除树干”分开处理。只有已经由多高度切片证明具有垂直连续性的细柱状点，才会作为树干从原始点云恢复到所属单木；恢复过程不会把地面、LAS 噪声分类点或远离树干的车辆点加入树木。")
    for item in [
        "保留：树干、主枝、树冠以及与树体连续的有效点。",
        "排除：地面分类点、低/高噪声分类点、孤立离群点、车辆和明显人工物体。",
        "有疑问：进入人工复核，使用俯视/侧视检查，不要仅凭一种视角定稿。",
    ]: add_bullet(doc,item)
    doc.add_picture(str(dbh_path),width=Inches(6.45)); doc.paragraphs[-1].alignment=WD_ALIGN_PARAGRAPH.CENTER
    p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER;add_text(p,"图 4-1　树高和胸径的正确高度基准",size=9)
    doc.add_page_break()

    doc.add_heading("5　单木人工复核、粘连树拆分与胸径", level=1)
    add_table(doc,["操作","使用方法","结果"],[
        ["选择树木","在点云或右侧列表点击树编号","高亮当前树并显示属性"],
        ["分开粘连树","选择含两棵树的编号，点击“分开粘连树”","按下部树干几何自动拆为两个编号；点不重不漏"],
        ["合并选中","多选属于同一棵树的块后合并","形成一个编号"],
        ["删除误分团块","删除车辆、设备或孤立误分块","从树木成员中排除，不删除原始证据"],
        ["区域套索删除","框选局部错误点","仅修改当前编辑版本"],
        ["撤销","撤回最近一次人工编辑","可检查后再保存"],
        ["胸径复核","选树后打开原始地面+树干 1.30 m 截面","显示拟合直径、质量和证据"],
        ["完成复核","定稿全部单木并生成数据","进入查看数据和预测"],
    ],widths=[3.0,7.0,5.8],font_size=8.7)
    doc.add_heading("5.1　树 04 为两棵相连树时", level=2)
    for n,title,body in [
        (1,"选择 T-1-004","确认高亮范围内确有两个独立树干。"),
        (2,"点击“分开粘连树”","软件利用靠近地面的树干中心进行确定性二分，不只按树冠等分。"),
        (3,"切换俯视与侧视","检查两个新编号的树干和树冠是否分别连续。"),
        (4,"必要时微调","用合并、删除误分团块、区域套索删除或撤销完成修正。"),
        (5,"分别胸径复核","选中每个新编号，检查原始地面 + 1.30 m 截面。"),
    ]: add_step(doc,n,title,body)
    add_note(doc,"胸径无法打开时", "先确认当前项目的分析状态为“完成”，并在左侧/点云中选中一棵树。无需先最终定稿；若是旧结果，先重新运行单木分割。", "FFF2CC")
    doc.add_page_break()

    doc.add_heading("6　查看数据、刷新与另存为", level=1)
    if SCREEN_DATA.exists():
        doc.add_picture(str(SCREEN_DATA),width=Inches(6.55));doc.paragraphs[-1].alignment=WD_ALIGN_PARAGRAPH.CENTER
        p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER;add_text(p,"图 6-1　查看数据表格与导出区域",size=9)
    add_para(doc,"“查看数据”显示当前项目全部已定稿单木，不再限制 10 棵。表格包含树高、胸径、冠幅、水平/垂直/三维净空、风险、质量和测量来源。")
    add_step(doc,1,"刷新数据","点击右上角“刷新数据”。按钮会显示忙碌状态；若失败会直接显示错误原因。")
    add_step(doc,2,"选择导出格式","在输出格式下拉框选择 Excel（.xlsx）、CSV、TSV、TXT 或 JSON。")
    add_step(doc,3,"导出当前项目数据","点击导出后弹出 Windows“另存为”，选择保存位置和文件名。")
    add_step(doc,4,"应用到预测","选中一行后进入树木预测；多棵预测请在预测页设置数量并按顺序选择编号。")
    add_note(doc,"格式对应", "输入支持的表格格式均提供对应输出格式。Excel 使用工作簿；CSV/TSV/TXT/JSON 为通用交换格式。", "E7F5EE")
    doc.add_page_break()

    doc.add_heading("7　树木预测（1～多棵）",level=1)
    doc.add_picture(str(pred_path),width=Inches(6.55));doc.paragraphs[-1].alignment=WD_ALIGN_PARAGRAPH.CENTER
    p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER;add_text(p,"图 7-1　统一预测入口的数据流",size=9)
    add_step(doc,1,"选择树木数量","输入要预测的棵数。选择 1 即原单木预测；选择 10 即出现 10 个按顺序排列的树编号选择框。")
    add_step(doc,2,"选择编号","每个位置选择一个不同的树编号，软件自动带入树高、胸径、冠幅、净空和导线信息。")
    add_step(doc,3,"填写树种与树龄","树种可从与模型训练一致的目录中选择，也可直接输入；自动识别仅作为建议，人工选择优先。")
    add_step(doc,4,"设置时间","填写观测起始日期和预测目标日期；系统按实际日历逐日积分，支持天、月和年。")
    add_step(doc,5,"天气与位置","选择手动输入或接入气象数据；经纬度可以手动输入并更新地图位置。")
    add_step(doc,6,"执行预测","软件逐树计算季节生长、净空变化、风偏和风险；列表按选择顺序输出。")
    add_step(doc,7,"查看每棵工单","结果行提供独立报告/工单入口，多棵树不会混成一个难以阅读的长段落。")
    doc.add_heading("7.1　预测结果必须关注的字段",level=2)
    add_table(doc,["字段","含义"],[
        ["当前/预测净空","树冠与导线在当前和目标日期的三维有效距离"],
        ["到达阈值日期与天数","预计进入用户设定危险范围的日期及距起始日天数"],
        ["预计接触风级","结合树种枝条响应、风向与净空估计可能接触的风级"],
        ["风险等级","根据当前自定义标准计算的分级结果"],
        ["证据质量","原始测量、导线拟合和胸径证据是否完整"],
    ],widths=[4.1,11.7])
    doc.add_page_break()

    doc.add_heading("8　树种、时间、季节生长和天气",level=1)
    doc.add_heading("8.1　树种与树龄",level=2)
    add_para(doc,"树种下拉目录与模型训练页共用同一主目录；如果目录中没有目标树种，可在同一输入框直接输入中文或学名。人工输入/选择的树种优先于自动识别结果。")
    doc.add_heading("8.2　按实际季节生长",level=2)
    add_para(doc,"孪生与预测不是等比例放大。系统按实际日期逐日计算：春季萌发和夏季旺长期增长较多，秋季缓生，冬季休眠期增长很少；树高、胸径和冠幅使用不同的生长分配，且受树种、树龄和天气修正。")
    add_note(doc,"科学边界","软件给出的是基于当前点云、树种参数和气象情景的工程预测，不等同于长期现场观测。极端修剪、病虫害、倒伏、施工和突发灾害需人工更新。", "FFF2CC")
    doc.add_heading("8.3　天气来源",level=2)
    add_table(doc,["模式","操作","适用场景"],[
        ["手动输入","点击手动天气，勾选/填写温度、湿度、降水、风速、风向、日照等需要项目","现场记录、离线运行、情景模拟"],
        ["气象数据接入","填写经纬度后匹配站点或导入气象局 JSON/CSV 文件","有网络或已有正式气象文件"],
        ["离线地图","不请求在线底图，行政边界和选点可在断网时使用","现场无网络"],
    ],widths=[3.0,7.5,5.3],font_size=8.7)
    add_para(doc,"纬度、经度输入框允许手工修改。输入后点击/触发更新，地图标记和气象站匹配都使用新坐标。")
    doc.add_page_break()

    doc.add_heading("9　标准设置与恢复默认",level=1)
    add_para(doc,"预测页面最上方显示当前标准摘要。点击“设置标准”可修改净空阈值、风险等级边界、权重等；点击“恢复默认”可恢复内置通用标准。")
    add_step(doc,1,"先设置标准","在预测前确认线路等级、净空标准、预警范围和权重。")
    add_step(doc,2,"保存标准","保存后当前项目立即使用新值，页面顶部摘要同步更新。")
    add_step(doc,3,"重新预测","标准发生变化后旧报告被标记为过期；必须重新执行预测。")
    add_step(doc,4,"生成工单","新工单读取刚保存的标准，不再固定使用通用标准。")
    add_step(doc,5,"恢复默认","需要回到内置标准时点击“恢复默认”，确认后重新预测。")
    add_note(doc,"为什么要重新预测", "阈值改变会影响风险等级、到达危险范围的日期、预计天数和工单建议。为避免旧结果与新标准混用，软件会主动使旧报告失效。", "E7F5EE")
    doc.add_page_break()

    doc.add_heading("10　点云孪生与时间轴",level=1)
    add_para(doc,"完成单木复核并至少执行一次预测后，“点云孪生”才可进入。孪生页独立显示选中树木、拟合导线、当前/未来点云和参数曲线。")
    add_step(doc,1,"进入条件","当前项目已有已定稿树木、导线结果和预测结果。")
    add_step(doc,2,"选择树木","从编号列表选择一棵；多棵预测结果也可逐棵打开孪生。")
    add_step(doc,3,"拖动时间轴","连续拖动底部滑块。界面使用预先计算的时间序列即时插值，不应每次重新打开窗口。")
    add_step(doc,4,"观察几何变化","点云按树高、冠幅、枝条方向和季节比例分别变化，不进行简单整体缩放。")
    add_step(doc,5,"查看曲线","孪生图下方同步显示净空、树高、胸径、冠幅等随时间变化的折线。")
    add_step(doc,6,"识别越界","自定义阈值以水平线标出；曲线越过阈值后变为红色，并给出首次越界日期。")
    add_step(doc,7,"风偏情景","调整风级/风向可查看风偏后的接触风险；预测区也显示预计接触风级。")
    add_note(doc,"五年仍未接触", "先检查当前净空方向、树种/树龄、日期、冠幅增长参数、导线坐标及自定义阈值。若基础点云中的树—线位置不正确，时间再长也不会得到可信接触结论。", "FFF2CC")
    doc.add_page_break()

    doc.add_heading("11　外部表格批量预测",level=1)
    add_para(doc,"外部表格批量用于没有当前点云项目、但已有结构化测量数据的任务。树木信息和树线信息在表格中输入；观测日期、经纬度、天气来源和标准在表格外统一设置。")
    add_table(doc,["表格字段","要求"],[
        ["树编号","必填且唯一"], ["树种","可选择或自由输入"], ["树高/胸径/冠幅","使用米/厘米对应表头单位"],
        ["电压等级/导线高度","用于选择标准和垂直关系"], ["水平/垂直/三维净空","至少提供可用于风险判断的有效距离"],
    ],widths=[5.0,10.8])
    for n,title,body in [
        (1,"选择或新增行","直接在表格中录入多棵树与树线数据。"),
        (2,"设置共同条件","在表格外设置日期、经纬度、手动天气或气象数据接入。"),
        (3,"解析并预览","检查字段、单位和空值。"),
        (4,"批量预测","按行顺序生成风险、阈值日期/天数和风级。"),
        (5,"导出","选择 Excel/CSV/TSV/TXT/JSON，在“另存为”窗口自选路径。"),
    ]: add_step(doc,n,title,body)
    doc.add_heading("11.1　模型训练",level=2)
    add_para(doc,"模型训练页用于使用历史样本改进树种生长或风险模型。训练页树种目录与预测页一致，也允许输入新树种。模型训练不是完成一次常规预测的必需步骤；没有经过验证的新模型不应直接替代公式模型用于正式结论。")
    doc.add_page_break()

    doc.add_heading("12　预测报告、工单与历史记录",level=1)
    add_para(doc,"工单采用与主界面统一的深色主题，按树编号排列，避免展示普通使用者难以理解的中间系数。核心内容如下。")
    add_table(doc,["区域","包含内容"],[
        ["任务对象","项目、线路/跨档、树编号、位置"],
        ["实际数据","当前树高、胸径、冠幅、当前净空、证据质量"],
        ["预测数据","目标日期树木参数、预测净空、风险等级"],
        ["阈值结论","预计多少天、预计哪天进入危险阈值范围"],
        ["风偏结论","预计达到几级风时可能接触，以及采用的风向条件"],
        ["处置建议","复测、修剪、巡检优先级和完成时限"],
    ],widths=[4.0,11.8])
    add_para(doc,"从工单返回后，当前项目和预测结果仍保存在项目目录；再次进入报告页会读取最近一次有效预测。修改标准或输入数据后需重新预测，旧工单不再作为当前结果。")
    doc.add_heading("12.1　导出与打印",level=2)
    add_bullet(doc,"数据表：使用“导出数据”并在另存为窗口选路径。")
    add_bullet(doc,"工单/报告：使用打印功能，可在 Windows 打印窗口选择“Microsoft Print to PDF”。")
    add_bullet(doc,"历史记录：用于查询以前生成的结果；正式归档建议同时保存项目文件夹和 PDF。")
    doc.add_page_break()

    doc.add_heading("13　常见问题与处理",level=1)
    add_table(doc,["现象","原因与处理"],[
        ["树干仍缺失/胸径为 0","旧项目结果未重算；重新运行单木分割。确认原始 LAS 包含地面和树干点。"],
        ["胸径复核提示不可用","确认分析已完成并已选择树木；无需先最终定稿。若无 1.30 m 有效截面，会显示质量原因而非伪造数值。"],
        ["两棵树仍是一个编号","选择该编号，点击“分开粘连树”；拆分后用俯视/侧视检查。"],
        ["查看数据刷新失败","检查当前项目是否完成定稿；保留错误提示并查看运行日志。"],
        ["导出没有弹窗","确认使用桌面 EXE 而不是单独浏览器页面；桌面版调用 Windows 另存为。"],
        ["点云孪生点不开","先完成复核并执行一次预测；检查所选树是否有有效导线/净空。"],
        ["工单仍显示旧标准","保存新标准后重新预测，再生成工单。"],
        ["离线地图空白","点击“不请求在线底图片”；离线行政边界仍可用。"],
        ["运行缓慢","大点云首次处理耗时较长；孪生拖动使用缓存。GPU 不可用时自动使用 CPU。"],
        ["软件被 Windows 阻止","从本说明列出的正式 EXE 或快捷方式启动；若安全提示出现，检查文件路径与发布来源后选择允许。"],
    ],widths=[4.8,11.0],font_size=8.6)
    doc.add_heading("13.1　日志位置",level=2)
    add_para(doc,r"运行日志统一保存在：D:\TreeLineWarning_一体化软件_20260926\TreeLineWarning一体化软件2.0_20260926\运行日志。出现错误时，请保留项目名、操作步骤、时间和日志文件。")
    doc.add_page_break()

    doc.add_heading("附录 A　顶部按钮速查",level=1)
    add_table(doc,["分组","按钮","用途"],[
        ["项目","新建项目 / 打开项目 / 导入 LAS/LAZ / 导入 GIS","建立、恢复和导入工程数据"],
        ["处理","点云预处理 / 地面点去除 / 导线重建 / 单木分割","完成点云到单木数据的处理链"],
        ["复核","单木人工复核 / 查看数据","编辑、胸径检查、定稿和数据导出"],
        ["预测","树木预测（1～多棵） / 外部表格批量 / 点云孪生 / 模型训练","当前项目预测、外部批量、时序可视化和模型管理"],
        ["输出","导出数据 / 预测报告与工单 / 历史记录","交换、打印与追溯结果"],
    ],widths=[2.2,7.3,6.4],font_size=8.7)
    doc.add_heading("附录 B　质量与验收说明",level=1)
    add_para(doc,"正式版已完成 Python 源码编译检查、JavaScript 语法检查、胸径树干恢复与粘连树拆分单元测试，以及应用测试集。PDAL、LAZ 读写、PROJ 离线坐标转换、WebView2 和 .NET 环境检查通过。未检测到可用 CUDA/CuPy 时，软件自动采用 CPU 计算。")
    add_note(doc,"结果责任", "自动结果必须结合现场复核、线路规程和具有资质人员判断。软件不会在证据不足时伪造安全结论。", "FCE8E6")

    # document properties
    doc.core_properties.title = "TreeLineWarning一体化软件2.0 全流程使用说明"
    doc.core_properties.subject = "点云处理、树木预测、点云孪生与预警工单操作手册"
    doc.core_properties.author = "TreeLineWarning 项目组"
    doc.core_properties.keywords = "TreeLineWarning, 点云, 胸径, 树木预测, 孪生, 预警"
    doc.save(DOCX_PATH)
    print(DOCX_PATH)


if __name__ == "__main__":
    build_document()
