"""Beijing tree catalog shared by warning and local species recognition.

The catalog focuses on trees and small trees that can occur in Beijing's urban,
plain, mountain, roadside, park and village landscapes.  It intentionally does
not copy every shrub from the official 140-name woody-plant list.
"""

# ruff: noqa: E501

from __future__ import annotations

from typing import Any

CATALOG_SOURCES = (
    {
        "title": "北京市主要乡土树种名录（2021版）",
        "organization": "北京市园林绿化局",
        "url": "https://yllhj.beijing.gov.cn/zwgk/sjfb/mlxx/202204/t20220418_2679623.shtml",
        "use": "北京乡土树种范围、生活型、适生区域和生态习性",
    },
    {
        "title": "北京市政协十三届四次会议第0142号提案答复",
        "organization": "北京市园林绿化局",
        "url": "https://yllhj.beijing.gov.cn/zwgk/tajydf/202110/t20211014_2512486.shtml",
        "use": "北京行道树实际应用范围",
    },
    {
        "title": "中国植物志／Flora of China",
        "organization": "中国科学院植物研究所等",
        "url": "https://www.iplant.cn/frps",
        "use": "中文名、学名和可人工复核的形态特征",
    },
    {
        "title": "城市公园树木安全风险评估——以北京动物园为例",
        "organization": "北京林业大学学报，2025，47(3):128-138",
        "url": "https://cjournal.hep.com.cn/1000-1522/CN/1160127524249461502",
        "use": "预警需结合树干、根系、偏冠和病虫害现场缺陷，不得仅按树种判定",
    },
)

# name: (scientific name, life form, concise field-identification cues, growth group)
# The cues are review prompts, not a replacement for keys, specimens or expert identification.
BEIJING_TREE_PROFILES: dict[str, tuple[str, str, str, str]] = {
    "法国梧桐": ("Platanus × acerifolia", "落叶乔木", "掌状大叶；球形果序；树皮片状剥落呈斑驳", "fast"),
    "国槐": ("Styphnolobium japonicum", "落叶乔木", "奇数羽状复叶；圆锥花序乳白花；念珠状荚果", "medium"),
    "龙爪槐": ("Styphnolobium japonicum f. pendula", "落叶小乔木", "国槐复叶和荚果；枝条明显下垂、伞形冠", "slow"),
    "白蜡": ("Fraxinus chinensis", "落叶乔木", "对生羽状复叶；小叶具锯齿；翅果成簇", "medium"),
    "绒毛白蜡": ("Fraxinus velutina", "落叶乔木", "对生羽状复叶；小枝和叶背常有绒毛；单翅果", "medium"),
    "大叶白蜡": ("Fraxinus rhynchophylla", "落叶乔木", "对生羽状复叶；小叶较宽大；翅果顶端钝或微凹", "medium"),
    "美国白蜡": ("Fraxinus pennsylvanica", "落叶乔木", "对生羽状复叶；叶背较淡；狭长单翅果", "medium"),
    "毛白杨": ("Populus tomentosa", "落叶乔木", "叶卵形或三角卵形；幼叶背白绒毛；老树基部深裂", "fast"),
    "杨树": ("Populus spp.", "落叶乔木组", "三角形至卵形叶；叶柄常侧扁；柔荑花序", "fast"),
    "新疆杨": ("Populus alba var. pyramidalis", "落叶乔木", "树冠窄柱形；叶背白色绒毛；枝条直立", "fast"),
    "小叶杨": ("Populus simonii", "落叶乔木", "叶菱状卵形且较小；树皮灰绿色；冠形开张", "fast"),
    "山杨": ("Populus davidiana", "落叶乔木", "近圆叶；叶柄侧扁使叶片易颤动；树皮灰绿", "fast"),
    "加拿大杨": ("Populus × canadensis", "落叶乔木", "三角叶较大；叶柄侧扁；速生高大、主干通直", "fast"),
    "旱柳": ("Salix matsudana", "落叶乔木", "狭披针形叶；细锯齿；枝条斜展而不明显下垂", "fast"),
    "垂柳": ("Salix babylonica", "落叶乔木", "长披针形叶；细长枝明显下垂；常见于水边", "fast"),
    "馒头柳": ("Salix matsudana f. umbraculifera", "落叶乔木", "叶似旱柳；枝密集向上；树冠半球形似馒头", "fast"),
    "绦柳": ("Salix matsudana f. pendula", "落叶乔木", "旱柳型叶；枝条长而下垂；冠较疏展", "fast"),
    "龙爪柳": ("Salix matsudana 'Tortuosa'", "落叶乔木", "枝条和小枝显著扭曲；狭披针形叶", "fast"),
    "榆树": ("Ulmus pumila", "落叶乔木", "叶基常不对称；叶缘重锯齿；近圆翅果种子居中", "medium"),
    "大果榆": ("Ulmus macrocarpa", "落叶小乔木", "叶较厚粗糙；翅果大且常有毛；耐旱山地型", "slow"),
    "脱皮榆": ("Ulmus lamellosa", "落叶乔木", "树皮薄片剥落呈斑驳；叶基偏斜；圆翅果", "slow"),
    "裂叶榆": ("Ulmus laciniata", "落叶乔木", "叶端常3—5裂；叶基不对称；翅果近圆", "medium"),
    "欧洲白榆": ("Ulmus laevis", "落叶乔木", "叶基强烈不对称；翅果边缘具毛；长果梗", "medium"),
    "金叶榆": ("Ulmus pumila 'Jinye'", "落叶小乔木", "榆树形叶；生长期叶色黄至金黄；常见嫁接冠", "medium"),
    "油松": ("Pinus tabuliformis", "常绿乔木", "针叶2针一束、较粗硬；卵圆球果；树皮灰褐鳞裂", "slow"),
    "白皮松": ("Pinus bungeana", "常绿乔木", "针叶3针一束；老树树皮片状脱落呈白绿斑驳", "slow"),
    "华山松": ("Pinus armandii", "常绿乔木", "针叶5针一束、柔软；球果大；种子无明显长翅", "medium"),
    "樟子松": ("Pinus sylvestris var. mongolica", "常绿乔木", "针叶2针一束；上部树皮橙红；球果小卵形", "medium"),
    "黑松": ("Pinus thunbergii", "常绿乔木", "针叶2针一束、粗硬深绿；冬芽银白；树皮黑灰", "medium"),
    "红松": ("Pinus koraiensis", "常绿乔木", "针叶5针一束；大球果；种子大且无翅", "medium"),
    "华北落叶松": ("Larix principis-rupprechtii", "落叶乔木", "针叶簇生、秋季金黄脱落；小球果宿存", "medium"),
    "落叶松": ("Larix spp.", "落叶乔木组", "针叶簇生且冬季脱落；枝上有短枝；球果直立宿存", "medium"),
    "白杄": ("Picea meyeri", "常绿乔木", "针叶四棱、粉蓝绿色；球果下垂；枝枕明显", "slow"),
    "青杄": ("Picea wilsonii", "常绿乔木", "针叶四棱、绿色；枝条较细；球果下垂", "slow"),
    "云杉": ("Picea spp.", "常绿乔木组", "单生四棱针叶着生于枝枕；球果下垂", "slow"),
    "雪松": ("Cedrus deodara", "常绿乔木", "针叶簇生；主梢和枝端常下垂；球果直立", "medium"),
    "冷杉": ("Abies spp.", "常绿乔木组", "扁平针叶、叶痕圆；球果直立且成熟时解体", "slow"),
    "水杉": ("Metasequoia glyptostroboides", "落叶乔木", "对生羽状小枝和线形叶；冬季整枝脱落；球果近球形", "fast"),
    "侧柏": ("Platycladus orientalis", "常绿乔木", "鳞叶小枝扁平直展；球果木质、种鳞顶端钩状", "slow"),
    "圆柏": ("Juniperus chinensis", "常绿乔木", "鳞叶与刺叶可同株；浆果状球果蓝黑；冠密", "slow"),
    "桧柏": ("Juniperus chinensis", "常绿乔木", "圆柏常用名；鳞叶与刺叶并存；浆果状球果", "slow"),
    "杜松": ("Juniperus rigida", "常绿小乔木", "针叶3枚轮生、硬尖；叶面有白色气孔带；球果浆果状", "slow"),
    "银杏": ("Ginkgo biloba", "落叶乔木", "扇形叶、二叉脉；长短枝明显；雌株种子黄色肉质外种皮", "slow"),
    "栾树": ("Koelreuteria paniculata", "落叶乔木", "羽状复叶；夏季黄色花；灯笼状蒴果", "medium"),
    "元宝枫": ("Acer truncatum", "落叶乔木", "掌状5裂叶；裂片全缘；双翅果张开近直角", "medium"),
    "五角枫": ("Acer pictum subsp. mono", "落叶乔木", "掌状5裂叶；叶基心形；双翅果近平展", "medium"),
    "茶条槭": ("Acer tataricum subsp. ginnala", "落叶小乔木", "叶常3裂、中裂片长；双翅果红色、近平展", "medium"),
    "三角枫": ("Acer buergerianum", "落叶乔木", "叶通常3浅裂；叶背粉白；双翅果锐角张开", "medium"),
    "复叶槭": ("Acer negundo", "落叶乔木", "槭属中少见羽状复叶；小枝绿色；双翅果成串", "fast"),
    "银白槭": ("Acer saccharinum", "落叶乔木", "掌状深裂叶；叶背银白；双翅果夹角较宽", "fast"),
    "臭椿": ("Ailanthus altissima", "落叶乔木", "大型羽状复叶；小叶基部有腺齿；翅果成大簇", "fast"),
    "香椿": ("Toona sinensis", "落叶乔木", "大型羽状复叶有香气；小叶近全缘；蒴果开裂留中轴", "fast"),
    "刺槐": ("Robinia pseudoacacia", "落叶乔木", "羽状复叶；托叶刺；白色总状花序；扁荚果", "fast"),
    "合欢": ("Albizia julibrissin", "落叶乔木", "二回羽状复叶；粉红丝状雄蕊花；扁平荚果", "fast"),
    "皂荚": ("Gleditsia sinensis", "落叶乔木", "枝干常有分枝粗刺；一至二回羽状复叶；长带状荚果", "medium"),
    "山皂荚": ("Gleditsia japonica", "落叶乔木", "枝干有刺；羽状复叶；荚果较薄而扭曲", "medium"),
    "山槐": ("Albizia kalkora", "落叶乔木", "二回羽状复叶；淡黄白头状花；扁荚果", "medium"),
    "楸树": ("Catalpa bungei", "落叶乔木", "大心形叶常轮生；白花有紫斑；细长蒴果", "fast"),
    "梓树": ("Catalpa ovata", "落叶乔木", "大心形叶；淡黄花有紫斑；细长蒴果", "fast"),
    "毛泡桐": ("Paulownia tomentosa", "落叶乔木", "大心形对生叶密被毛；紫色漏斗花；卵形蒴果", "fast"),
    "泡桐": ("Paulownia spp.", "落叶乔木组", "大心形对生叶；紫色筒状花；木质蒴果", "fast"),
    "杜仲": ("Eucommia ulmoides", "落叶乔木", "单叶互生；折断叶或树皮可见白色胶丝；翅果", "medium"),
    "七叶树": ("Aesculus chinensis", "落叶乔木", "掌状复叶常7小叶；直立圆锥花序；蒴果近球形", "medium"),
    "构树": ("Broussonetia papyrifera", "落叶乔木", "叶形变化大且粗糙；乳汁；橙红聚花果", "fast"),
    "桑树": ("Morus alba", "落叶乔木", "叶常有裂、基部心形；乳汁；聚花果成熟紫黑或白", "fast"),
    "枣树": ("Ziziphus jujuba", "落叶小乔木", "叶基三出脉；托叶刺；核果红褐；小枝曲折", "medium"),
    "核桃": ("Juglans regia", "落叶乔木", "奇数羽状复叶且有香气；核果外果皮绿色；芽叠生", "medium"),
    "核桃楸": ("Juglans mandshurica", "落叶乔木", "大型羽状复叶；叶轴有毛；果实卵形、果核深沟", "medium"),
    "白桦": ("Betula platyphylla", "落叶乔木", "白色纸状树皮横向剥落；三角卵形叶；果序圆柱形", "medium"),
    "鹅耳枥": ("Carpinus turczaninowii", "落叶乔木", "叶脉平行明显、重锯齿；果苞叶状成串", "slow"),
    "板栗": ("Castanea mollissima", "落叶乔木", "长椭圆叶具芒状锯齿；雄花序长；壳斗密生长刺", "medium"),
    "麻栎": ("Quercus acutissima", "落叶乔木", "叶缘芒状锯齿；壳斗鳞片长而反曲；橡实卵形", "medium"),
    "栓皮栎": ("Quercus variabilis", "落叶乔木", "树皮厚软栓质；叶背灰白；壳斗鳞片反曲", "medium"),
    "槲树": ("Quercus dentata", "落叶乔木", "叶大、倒卵形且浅裂；嫩枝叶有毛；壳斗杯状", "slow"),
    "辽东栎": ("Quercus mongolica subsp. crispula", "落叶乔木", "叶倒卵形、羽状浅裂；叶柄短；橡实椭圆", "slow"),
    "蒙古栎": ("Quercus mongolica", "落叶乔木", "叶倒卵形、羽状裂；叶基耳形；壳斗杯状", "slow"),
    "夏橡": ("Quercus robur", "落叶乔木", "叶柄极短、叶基耳状；橡实具明显长果梗", "medium"),
    "青檀": ("Pteroceltis tatarinowii", "落叶乔木", "树皮灰色片状剥落；叶基不对称；小坚果具翅", "slow"),
    "朴树": ("Celtis sinensis", "落叶乔木", "叶基三出脉且略偏斜；上半部具齿；核果近球形", "medium"),
    "小叶朴": ("Celtis bungeana", "落叶乔木", "叶较小、基部三出脉；叶缘上部有锯齿；核果橙红", "medium"),
    "黄连木": ("Pistacia chinensis", "落叶乔木", "偶数羽状复叶；秋叶红；小核果由红转蓝黑", "medium"),
    "黄栌": ("Cotinus coggygria", "落叶小乔木", "圆卵形全缘叶；果序羽毛状；秋叶鲜红", "medium"),
    "火炬树": ("Rhus typhina", "落叶小乔木", "羽状复叶；小枝密毛；红色圆锥果序直立宿存", "fast"),
    "盐肤木": ("Rhus chinensis", "落叶小乔木", "羽状复叶、叶轴具翅；果序密集；常见虫瘿", "medium"),
    "柿树": ("Diospyros kaki", "落叶乔木", "叶厚亮全缘；萼片宿存且大；橙黄色浆果", "medium"),
    "君迁子": ("Diospyros lotus", "落叶乔木", "叶较小；果实小、熟时蓝黑；萼片宿存", "medium"),
    "椴树": ("Tilia spp.", "落叶乔木组", "心形锯齿叶；花序连舌状苞片；小坚果", "medium"),
    "蒙椴": ("Tilia mongolica", "落叶乔木", "叶常3—5浅裂；花序具舌状苞片；果有棱", "medium"),
    "糠椴": ("Tilia mandshurica", "落叶乔木", "大心形叶、叶背星状毛；花序具舌状苞片", "medium"),
    "流苏树": ("Chionanthus retusus", "落叶小乔木", "对生全缘叶；白色细长花瓣成流苏状；蓝黑核果", "slow"),
    "暴马丁香": ("Syringa reticulata", "落叶小乔木", "对生卵形叶；大型白色圆锥花序；树皮横裂", "medium"),
    "杜梨": ("Pyrus betulifolia", "落叶乔木", "卵形锯齿叶；白花；小梨果褐色、宿存萼脱落", "medium"),
    "秋子梨": ("Pyrus ussuriensis", "落叶乔木", "叶缘锐锯齿；白花；小梨果黄绿色", "medium"),
    "山楂": ("Crataegus pinnatifida", "落叶小乔木", "叶羽状深裂；枝可有刺；红色梨果", "medium"),
    "山桃": ("Prunus davidiana", "落叶小乔木", "早春粉白花先叶开放；叶披针形；核果近球形", "medium"),
    "山杏": ("Prunus sibirica", "落叶小乔木", "早春白或粉花先叶；圆卵形叶；果肉薄、核粗糙", "medium"),
    "杏树": ("Prunus armeniaca", "落叶小乔木", "圆卵形叶、基部心形；花先叶；橙黄核果", "medium"),
    "桃树": ("Prunus persica", "落叶小乔木", "狭披针形叶；花先叶；核果表面多有绒毛", "medium"),
    "碧桃": ("Prunus persica Duplex Group", "落叶小乔木", "桃形狭叶；重瓣观赏花；果实少或小", "medium"),
    "紫叶李": ("Prunus cerasifera 'Atropurpurea'", "落叶小乔木", "叶全年紫红；早春粉白花；小核果", "medium"),
    "稠李": ("Padus avium", "落叶乔木", "椭圆锯齿叶；白花成总状花序；小核果黑色", "medium"),
    "樱花": ("Cerasus spp.", "落叶乔木组", "叶柄顶端常有腺体；花簇生；树皮有横向皮孔", "medium"),
    "山樱花": ("Cerasus serrulata", "落叶乔木", "叶尖长、锯齿芒尖；花成伞房状；树皮横纹", "medium"),
    "海棠": ("Malus spectabilis", "落叶小乔木", "花蕾红、开放粉白；椭圆锯齿叶；小梨果黄色", "medium"),
    "西府海棠": ("Malus × micromalus", "落叶小乔木", "花蕾深红、花粉红；果较小红黄、果梗长", "medium"),
    "苹果": ("Malus domestica", "落叶小乔木", "叶背有毛；花白至粉红；大型梨果", "medium"),
    "花楸": ("Sorbus pohuashanensis", "落叶乔木", "羽状复叶；白花成复伞房；红色小梨果成簇", "medium"),
    "玉兰": ("Yulania denudata", "落叶乔木", "早春大型白花先叶；冬芽密被绒毛；聚合蓇葖果", "medium"),
    "二乔玉兰": ("Yulania × soulangeana", "落叶小乔木", "早春大型杯状花、外侧紫红内侧白；花先叶", "medium"),
    "木兰": ("Magnolia spp.", "落叶乔木组", "单叶全缘；大型单生花；聚合蓇葖果", "medium"),
    "苦楝": ("Melia azedarach", "落叶乔木", "二至三回羽状复叶；淡紫花；黄色核果冬季宿存", "fast"),
    "文冠果": ("Xanthoceras sorbifolium", "落叶小乔木", "羽状复叶；白花基部由黄转红；厚壁蒴果", "medium"),
    "丝棉木": ("Euonymus maackii", "落叶小乔木", "对生细锯齿叶；四裂粉红蒴果；橙红假种皮", "medium"),
    "山茱萸": ("Cornus officinalis", "落叶小乔木", "对生全缘叶、弧形侧脉；黄花先叶；红色核果", "slow"),
    "紫薇": ("Lagerstroemia indica", "落叶小乔木", "树皮光滑片状脱落；圆锥花序；蒴果宿存", "medium"),
    "石榴": ("Punica granatum", "落叶小乔木", "小枝常有刺；叶对生全缘；红花、浆果状果实", "medium"),
    "灯台树": ("Cornus controversa", "落叶乔木", "侧枝层状轮生似灯台；叶互生、弧形脉；蓝黑核果", "medium"),
    "毛梾": ("Cornus walteri", "落叶乔木", "对生叶、叶背有毛；白花成伞房；黑色核果", "medium"),
    "其他": ("Unresolved tree taxon", "待鉴定乔木", "请记录叶序、叶形、树皮、花果、冬芽和整体冠形并请专业人员复核", "default"),
}

BEIJING_TREE_SPECIES = tuple(BEIJING_TREE_PROFILES)

GROWTH_GROUPS: dict[str, dict[str, float | str]] = {
    "fast": {"minimum": 0.9, "maximum": 1.8, "reference": 1.2, "source": "北京树种目录习性分组工程初值"},
    "medium": {"minimum": 0.45, "maximum": 1.0, "reference": 0.7, "source": "北京树种目录习性分组工程初值"},
    "slow": {"minimum": 0.2, "maximum": 0.65, "reference": 0.4, "source": "北京树种目录习性分组工程初值"},
    "default": {"minimum": 0.55, "maximum": 0.85, "reference": 0.7, "source": "未鉴定乔木通用工程初值"},
}


def species_profile(name: str) -> dict[str, Any]:
    scientific_name, life_form, features, growth_group = BEIJING_TREE_PROFILES.get(
        name, BEIJING_TREE_PROFILES["其他"]
    )
    return {
        "name": name,
        "scientific_name": scientific_name,
        "life_form": life_form,
        "identification_features": features,
        "growth_group": growth_group,
        "growth_reference": dict(GROWTH_GROUPS[growth_group]),
    }


def catalog_payload() -> dict[str, Any]:
    return {
        "count": len(BEIJING_TREE_SPECIES) - 1,
        "species": [species_profile(name) for name in BEIJING_TREE_SPECIES],
        "sources": list(CATALOG_SOURCES),
        "scope": "北京常见城市、平原、山区、道路、公园和村镇乔木/小乔木；不含大多数灌木和藤本",
        "warning": "形态特征仅供复核；自动识别必须使用本地标注样本训练，预警仍须结合净空、树体缺陷、气象和现场测量。",
    }
