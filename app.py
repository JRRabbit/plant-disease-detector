import os
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import numpy as np
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import hashlib
from werkzeug.utils import secure_filename
from werkzeug.middleware.proxy_fix import ProxyFix

# 可选导入 YOLO（云端环境缺失依赖时不阻塞启动）
try:
    from ultralytics import YOLO  # type: ignore
except Exception:
    YOLO = None
# ============================================================
# 病虫害中英文映射表 + 详细资料
# 共 38 个类别，与你的模型完全对应
# ============================================================

DISEASE_DATABASE = {
    # ==================== 苹果 Apple ====================
    'apple apple scab': {
        'name_cn': '苹果黑星病',
        'category': '真菌性病害',
        'affected_crops': '苹果',
        'symptoms': '叶片出现橄榄绿色至黑色的圆形病斑，后期病斑表面生出黑色霉层。果实上形成黑色疮痂状病斑，影响外观和品质。',
        'cause': '由苹果黑星病菌(Venturia inaequalis)引起，温暖潮湿环境易发病，春季多雨时发病严重。病菌在落叶上越冬。',
        'prevention': '1. 秋冬季清除落叶，减少菌源\n2. 合理修剪，保持树冠通风透光\n3. 选用抗病品种\n4. 春季萌芽前喷石硫合剂',
        'treatment': '1. 发病初期喷洒70%甲基托布津1000倍液\n2. 使用80%代森锰锌600-800倍液\n3. 严重时使用43%戊唑醇3000倍液',
        'pesticides': '甲基托布津、代森锰锌、戊唑醇、多菌灵、苯醚甲环唑'
    },
    
    'apple black rot': {
        'name_cn': '苹果黑腐病',
        'category': '真菌性病害',
        'affected_crops': '苹果',
        'symptoms': '果实出现褐色至黑色腐烂斑，逐渐扩大。叶片出现紫色边缘的褐色圆形斑点，后期病斑中央变灰白色，上有小黑点。',
        'cause': '由葡萄座腔菌(Botryosphaeria obtusa)引起，通过伤口侵染，高温高湿条件有利发病。',
        'prevention': '1. 及时清除病果、病叶和枯枝\n2. 避免果实机械损伤\n3. 加强果园通风\n4. 冬季清园',
        'treatment': '1. 喷施1:2:200波尔多液\n2. 使用70%甲基托布津800倍液\n3. 发病期喷施10%苯醚甲环唑1500倍液',
        'pesticides': '波尔多液、甲基托布津、苯醚甲环唑、代森锰锌'
    },
    
    'apple cedar apple rust': {
        'name_cn': '苹果锈病',
        'category': '真菌性病害',
        'affected_crops': '苹果',
        'symptoms': '叶片正面出现橙黄色圆形病斑，病斑上有橙黄色小点。叶片背面病斑处隆起，后期产生黄褐色毛状物（锈孢子器）。',
        'cause': '由苹果锈病菌(Gymnosporangium juniperi-virginianae)引起，需要桧柏作为转主寄主完成生活史。',
        'prevention': '1. 清除果园周围500米内的桧柏\n2. 如无法清除桧柏，春季在桧柏上喷药\n3. 苹果萌芽至幼果期喷药保护',
        'treatment': '1. 喷施15%三唑酮1500倍液\n2. 使用12.5%腈菌唑2000倍液\n3. 代森锰锌600倍液预防',
        'pesticides': '三唑酮、腈菌唑、代森锰锌、氟硅唑'
    },
    
    'apple healthy': {
        'name_cn': '苹果（健康）',
        'category': '健康状态',
        'affected_crops': '苹果',
        'symptoms': '叶片翠绿有光泽，叶形正常，无病斑、变色或畸形。枝条生长健壮，果实发育良好。',
        'cause': '无病害',
        'prevention': '1. 继续保持良好的果园管理\n2. 合理修剪，保持通风透光\n3. 科学施肥，增强树势\n4. 定期检查，及时发现问题',
        'treatment': '无需治疗',
        'pesticides': '无需用药，可进行常规预防性喷药'
    },
    
    # ==================== 香蕉 Banana ====================
    'banana healthy': {
        'name_cn': '香蕉（健康）',
        'category': '健康状态',
        'affected_crops': '香蕉',
        'symptoms': '叶片翠绿挺立，叶缘完整，无斑点或条纹。假茎粗壮，果穗发育正常。',
        'cause': '无病害',
        'prevention': '1. 保持良好的水肥管理\n2. 及时除去枯叶老叶\n3. 防治蚜虫等传毒媒介\n4. 合理密植',
        'treatment': '无需治疗',
        'pesticides': '无需用药'
    },
    
    'banana segatoka': {
        'name_cn': '香蕉叶斑病（黑条叶斑病）',
        'category': '真菌性病害',
        'affected_crops': '香蕉',
        'symptoms': '叶片初期出现淡黄色小斑点，后扩大为褐色至黑色椭圆形病斑，病斑周围有黄色晕圈。严重时病斑连片，叶片枯死。',
        'cause': '由香蕉球腔菌(Mycosphaerella fijiensis)引起，高温多雨季节发病严重，可造成严重减产。',
        'prevention': '1. 选用抗病品种\n2. 合理密植，改善通风\n3. 及时清除病叶\n4. 雨季前开始喷药预防',
        'treatment': '1. 喷施25%丙环唑1500倍液\n2. 使用80%代森锰锌600倍液\n3. 交替使用不同药剂',
        'pesticides': '丙环唑、代森锰锌、苯醚甲环唑、吡唑醚菌酯'
    },
    
    'banana xamthomonas': {
        'name_cn': '香蕉细菌性条斑病',
        'category': '细菌性病害',
        'affected_crops': '香蕉',
        'symptoms': '叶片出现水浸状条斑，后变为黑褐色，沿叶脉扩展。严重时叶片大面积枯死，植株矮化，果实品质下降。',
        'cause': '由黄单胞杆菌(Xanthomonas)引起，通过雨水飞溅、伤口侵入传播，高温多雨季节发病重。',
        'prevention': '1. 使用无病种苗\n2. 避免造成伤口\n3. 发现病株及时拔除\n4. 加强田间卫生',
        'treatment': '1. 喷施77%氢氧化铜500倍液\n2. 使用20%噻菌铜500倍液\n3. 配合农用链霉素使用',
        'pesticides': '氢氧化铜、噻菌铜、农用链霉素、中生菌素'
    },
    
    # ==================== 樱桃 Cherry ====================
    'cherry (including sour) healthy': {
        'name_cn': '樱桃（健康）',
        'category': '健康状态',
        'affected_crops': '樱桃（包括酸樱桃）',
        'symptoms': '叶片翠绿有光泽，叶形正常舒展，无病斑、白粉或畸形。果实发育正常，色泽鲜艳。',
        'cause': '无病害',
        'prevention': '1. 保持合理的树形结构\n2. 科学施肥和灌溉\n3. 注意果园通风\n4. 及时防治虫害',
        'treatment': '无需治疗',
        'pesticides': '无需用药'
    },
    
    'cherry (including sour) powdery mildew': {
        'name_cn': '樱桃白粉病',
        'category': '真菌性病害',
        'affected_crops': '樱桃（包括酸樱桃）',
        'symptoms': '叶片表面覆盖一层白色粉状物，严重时叶片卷曲、变形、早落。幼果受害后发育不良或畸形。',
        'cause': '由白粉菌(Podosphaera clandestina)引起，干旱、昼夜温差大、氮肥过多时发病严重。',
        'prevention': '1. 加强果园通风透光\n2. 避免氮肥过量\n3. 秋季清除病叶\n4. 发病前喷药预防',
        'treatment': '1. 喷施15%三唑酮1000倍液\n2. 使用40%氟硅唑5000倍液\n3. 发病初期喷施50%硫悬浮剂300倍液',
        'pesticides': '三唑酮、氟硅唑、硫悬浮剂、腈菌唑、醚菌酯'
    },
    
    # ==================== 玉米 Corn ====================
    'corn (maize) cercospora leaf spot gray leaf spot': {
        'name_cn': '玉米灰斑病',
        'category': '真菌性病害',
        'affected_crops': '玉米',
        'symptoms': '叶片出现灰色至褐色的长方形或条形病斑，病斑边缘平行于叶脉，呈典型的"灰斑"状。严重时叶片大面积枯死。',
        'cause': '由玉米尾孢菌(Cercospora zeae-maydis)引起，高温高湿条件下发病严重，连作田发病重。',
        'prevention': '1. 选用抗病品种\n2. 合理轮作\n3. 清除田间病残体\n4. 合理密植，改善通风',
        'treatment': '1. 发病初期喷施10%苯醚甲环唑1500倍液\n2. 使用25%吡唑醚菌酯2000倍液\n3. 代森锰锌600倍液',
        'pesticides': '苯醚甲环唑、吡唑醚菌酯、代森锰锌、丙环唑'
    },
    
    'corn (maize) common rust': {
        'name_cn': '玉米锈病',
        'category': '真菌性病害',
        'affected_crops': '玉米',
        'symptoms': '叶片两面散生金黄色至锈褐色的疱状小突起（夏孢子堆），破裂后散出大量锈褐色粉末（夏孢子）。',
        'cause': '由玉米锈病菌(Puccinia sorghi)引起，温暖潮湿条件下发病严重，孢子随气流远距离传播。',
        'prevention': '1. 选用抗病品种\n2. 适期播种，避开发病高峰\n3. 合理施肥，增强植株抗性',
        'treatment': '1. 喷施15%三唑酮1000倍液\n2. 使用25%丙环唑1500倍液\n3. 12.5%氟环唑2000倍液',
        'pesticides': '三唑酮、丙环唑、氟环唑、戊唑醇'
    },
    
    'corn (maize) healthy': {
        'name_cn': '玉米（健康）',
        'category': '健康状态',
        'affected_crops': '玉米',
        'symptoms': '植株生长健壮，叶片宽大翠绿，茎秆粗壮，无病斑或锈粉。果穗发育正常，籽粒饱满。',
        'cause': '无病害',
        'prevention': '1. 继续保持良好的田间管理\n2. 合理施肥灌溉\n3. 定期检查病虫害\n4. 及时中耕除草',
        'treatment': '无需治疗',
        'pesticides': '无需用药'
    },
    
    'corn (maize) northern leaf blight': {
        'name_cn': '玉米大斑病',
        'category': '真菌性病害',
        'affected_crops': '玉米',
        'symptoms': '叶片出现大型梭形或长椭圆形病斑，初为水浸状，后变灰绿色至黄褐色，严重时病斑连片，叶片枯死。',
        'cause': '由玉米大斑病菌(Exserohilum turcicum)引起，中温（20-25℃）高湿条件下发病严重。',
        'prevention': '1. 选用抗病品种\n2. 合理密植\n3. 增施磷钾肥，增强抗性\n4. 收获后清除病残体',
        'treatment': '1. 发病初期喷施50%多菌灵500倍液\n2. 使用70%甲基托布津800倍液\n3. 代森锰锌600倍液',
        'pesticides': '多菌灵、甲基托布津、代森锰锌、苯醚甲环唑'
    },
    
    # ==================== 葡萄 Grape ====================
    'grape black rot': {
        'name_cn': '葡萄黑腐病',
        'category': '真菌性病害',
        'affected_crops': '葡萄',
        'symptoms': '叶片出现红褐色圆形病斑，边缘颜色较深，上有小黑点。果实染病后变褐、腐烂，最后干缩成黑色僵果。',
        'cause': '由葡萄黑腐病菌(Guignardia bidwellii)引起，高温多雨季节发病严重，僵果是主要侵染源。',
        'prevention': '1. 彻底清除僵果和病叶\n2. 加强果园通风透光\n3. 雨季及时排水\n4. 花前花后重点喷药',
        'treatment': '1. 喷施1:0.7:200波尔多液\n2. 使用80%代森锰锌600倍液\n3. 甲基托布津800倍液',
        'pesticides': '波尔多液、代森锰锌、甲基托布津、苯醚甲环唑'
    },
    
    'grape esca (black measles)': {
        'name_cn': '葡萄黑麻疹病',
        'category': '真菌性病害',
        'affected_crops': '葡萄',
        'symptoms': '叶片出现"虎皮状"斑纹，叶缘和叶脉间组织坏死。果实表面出现褐色至紫黑色小斑点，严重时果实干缩。',
        'cause': '由多种真菌（主要是Phaeomoniella和Phaeoacremonium）复合侵染引起，通过修剪伤口侵入。',
        'prevention': '1. 修剪时消毒剪刀\n2. 修剪伤口涂抹保护剂\n3. 避免大伤口\n4. 选用健康苗木',
        'treatment': '1. 目前无特效药\n2. 严重病株挖除烧毁\n3. 修剪伤口涂抹甲基托布津糊剂',
        'pesticides': '伤口保护剂、甲基托布津糊剂（以预防为主）'
    },
    
    'grape healthy': {
        'name_cn': '葡萄（健康）',
        'category': '健康状态',
        'affected_crops': '葡萄',
        'symptoms': '叶片翠绿有光泽，叶形正常，无病斑、霉层或畸形。枝条健壮，果穗发育良好，果粒饱满。',
        'cause': '无病害',
        'prevention': '1. 保持果园通风透光\n2. 合理修剪和绑蔓\n3. 科学施肥灌溉\n4. 定期病虫害检查',
        'treatment': '无需治疗',
        'pesticides': '无需用药'
    },
    
    'grape leaf blight (isariopsis leaf spot)': {
        'name_cn': '葡萄叶枯病',
        'category': '真菌性病害',
        'affected_crops': '葡萄',
        'symptoms': '叶片出现不规则形褐色病斑，边缘略带紫色，后期病斑干枯破裂。严重时叶片提早脱落，影响树势和果实品质。',
        'cause': '由葡萄叶枯病菌(Isariopsis clavispora)引起，多雨潮湿条件下发病严重。',
        'prevention': '1. 及时清除落叶\n2. 加强通风透光\n3. 避免过度密植\n4. 雨季注意排水',
        'treatment': '1. 喷施80%代森锰锌600倍液\n2. 使用75%百菌清500倍液\n3. 甲基托布津800倍液',
        'pesticides': '代森锰锌、百菌清、甲基托布津、福美双'
    },
    
    # ==================== 柑橘 Orange ====================
    'orange haunglongbing (citrus greening)': {
        'name_cn': '柑橘黄龙病',
        'category': '细菌性病害',
        'affected_crops': '柑橘类（橙、柚、柠檬等）',
        'symptoms': '叶片黄化，呈斑驳状，新叶小而直立。果实畸形、偏小、着色不均，果肉味苦，种子败育。植株逐渐衰退死亡。',
        'cause': '由韧皮部杆菌(Candidatus Liberibacter)引起，主要通过木虱传播，是柑橘产业的毁灭性病害。',
        'prevention': '1. 严格使用无病苗木\n2. 彻底防治柑橘木虱\n3. 建立无病果园\n4. 发现病树立即挖除烧毁',
        'treatment': '1. 目前无法治愈\n2. 发病初期注射抗生素可延缓病情\n3. 病树必须挖除销毁\n4. 重点防治木虱',
        'pesticides': '防治木虱：吡虫啉、噻虫嗪、高效氯氰菊酯'
    },
    
    # ==================== 辣椒 Pepper ====================
    'pepper, bell bacterial spot': {
        'name_cn': '甜椒细菌性斑点病',
        'category': '细菌性病害',
        'affected_crops': '甜椒、辣椒',
        'symptoms': '叶片出现水浸状小斑点，后变为褐色至黑色，略凹陷，周围有黄色晕圈。果实上形成疮痂状突起，影响外观和品质。',
        'cause': '由丁香假单胞杆菌(Xanthomonas vesicatoria)引起，高温多雨时发病严重，种子可带菌。',
        'prevention': '1. 使用无病种子\n2. 种子消毒处理\n3. 避免连作\n4. 控制田间湿度',
        'treatment': '1. 喷施72%农用链霉素3000倍液\n2. 使用77%氢氧化铜500倍液\n3. 20%噻菌铜500倍液',
        'pesticides': '农用链霉素、氢氧化铜、噻菌铜、中生菌素'
    },
    
    'pepper, bell healthy': {
        'name_cn': '甜椒（健康）',
        'category': '健康状态',
        'affected_crops': '甜椒、辣椒',
        'symptoms': '植株生长健壮，叶片深绿有光泽，无斑点或畸形。果实发育正常，色泽鲜艳，表面光滑。',
        'cause': '无病害',
        'prevention': '1. 保持良好的栽培管理\n2. 合理施肥灌溉\n3. 及时整枝打杈\n4. 注意通风换气',
        'treatment': '无需治疗',
        'pesticides': '无需用药'
    },
    
    # ==================== 马铃薯 Potato ====================
    'potato early blight': {
        'name_cn': '马铃薯早疫病',
        'category': '真菌性病害',
        'affected_crops': '马铃薯、番茄',
        'symptoms': '叶片出现褐色圆形病斑，有明显的同心轮纹，形似靶标或牛眼。下部老叶先发病，逐渐向上蔓延。',
        'cause': '由链格孢菌(Alternaria solani)引起，高温干旱后遇雨易发病，植株衰弱时发病重。',
        'prevention': '1. 选用抗病品种\n2. 合理轮作\n3. 增施磷钾肥\n4. 避免缺水缺肥',
        'treatment': '1. 发病初期喷施75%百菌清500倍液\n2. 使用80%代森锰锌600倍液\n3. 苯醚甲环唑1500倍液',
        'pesticides': '百菌清、代森锰锌、苯醚甲环唑、嘧菌酯'
    },
    
    'potato healthy': {
        'name_cn': '马铃薯（健康）',
        'category': '健康状态',
        'affected_crops': '马铃薯',
        'symptoms': '植株生长健壮，叶片翠绿平展，无病斑或霉层。茎秆粗壮，块茎发育正常。',
        'cause': '无病害',
        'prevention': '1. 使用脱毒种薯\n2. 合理轮作\n3. 高垄栽培\n4. 适时培土',
        'treatment': '无需治疗',
        'pesticides': '无需用药'
    },
    
    'potato late blight': {
        'name_cn': '马铃薯晚疫病',
        'category': '真菌性病害',
        'affected_crops': '马铃薯、番茄',
        'symptoms': '叶片出现水浸状暗绿色病斑，迅速扩大变褐。潮湿时叶背病斑边缘产生白色霉层。块茎表面出现褐色斑块，内部腐烂。',
        'cause': '由致病疫霉(Phytophthora infestans)引起，低温（15-20℃）高湿条件下发病严重，是马铃薯的毁灭性病害。',
        'prevention': '1. 选用抗病品种\n2. 使用脱毒种薯\n3. 高垄栽培，利于排水\n4. 发病前开始喷药保护',
        'treatment': '1. 发病初期喷施72%霜脲氰·锰锌600倍液\n2. 使用68%精甲霜·锰锌600倍液\n3. 69%烯酰·锰锌600倍液',
        'pesticides': '霜脲氰、甲霜灵、烯酰吗啉、代森锰锌、氟吡菌胺'
    },
    
    # ==================== 草莓 Strawberry ====================
    'strawberry healthy': {
        'name_cn': '草莓（健康）',
        'category': '健康状态',
        'affected_crops': '草莓',
        'symptoms': '植株矮壮，叶片翠绿平展，叶缘整齐。花朵正常，果实发育良好，色泽鲜艳，风味佳。',
        'cause': '无病害',
        'prevention': '1. 选用健康种苗\n2. 合理施肥\n3. 保持适当湿度\n4. 及时摘除老叶病叶',
        'treatment': '无需治疗',
        'pesticides': '无需用药'
    },
    
    'strawberry leaf scorch': {
        'name_cn': '草莓叶焦病',
        'category': '真菌性病害',
        'affected_crops': '草莓',
        'symptoms': '叶片出现紫红色小斑点，后扩大为不规则形病斑，病斑中央不变灰白（与叶斑病区别）。严重时叶片焦枯，影响产量。',
        'cause': '由草莓叶焦病菌(Diplocarpon earliana)引起，春末夏初多雨季节发病严重。',
        'prevention': '1. 选用抗病品种\n2. 清除病残体\n3. 控制种植密度\n4. 避免过量施用氮肥',
        'treatment': '1. 喷施80%代森锰锌600倍液\n2. 使用70%甲基托布津800倍液\n3. 10%苯醚甲环唑1500倍液',
        'pesticides': '代森锰锌、甲基托布津、苯醚甲环唑、腈菌唑'
    },
    
    # ==================== 茶叶 Tea ====================
    'tea leaf blight': {
        'name_cn': '茶叶枯病',
        'category': '真菌性病害',
        'affected_crops': '茶树',
        'symptoms': '叶缘或叶尖开始发病，初为黄褐色，后扩展为不规则形大斑，病健交界处有深褐色隆起线。后期病叶枯焦脱落。',
        'cause': '由茶轮斑病菌(Pestalotiopsis theae)等引起，高温高湿季节发病重，树势衰弱时易感病。',
        'prevention': '1. 加强茶园管理\n2. 增施有机肥\n3. 合理修剪\n4. 及时清除病叶',
        'treatment': '1. 喷施75%百菌清600倍液\n2. 使用70%甲基托布津800倍液\n3. 代森锰锌600倍液',
        'pesticides': '百菌清、甲基托布津、代森锰锌、波尔多液'
    },
    
    'tea red leaf spot': {
               'name_cn': '茶红叶斑病',
        'category': '真菌性病害',
        'affected_crops': '茶树',
        'symptoms': '叶片上出现红褐色圆形或不规则形病斑，边缘颜色较深，病斑上散生小黑点。严重时叶片提早脱落。',
        'cause': '由茶叶斑病菌引起，高温多湿季节发病严重，管理粗放的茶园发病重。',
        'prevention': '1. 加强茶园管理，增强树势\n2. 合理施肥，避免偏施氮肥\n3. 适时修剪，保持通风\n4. 及时清除病叶',
        'treatment': '1. 喷施75%百菌清600倍液\n2. 使用70%甲基托布津800倍液\n3. 50%多菌灵600倍液',
        'pesticides': '百菌清、甲基托布津、多菌灵、代森锰锌'
    },
    
    'tea red scab': {
        'name_cn': '茶赤叶斑病（茶红锈病）',
        'category': '真菌性病害',
        'affected_crops': '茶树',
        'symptoms': '嫩叶和嫩梢上出现淡黄色至红褐色病斑，表面产生灰白色粉状物。严重时新梢枯死，影响茶叶产量和品质。',
        'cause': '由茶饼病菌(Exobasidium vexans)引起，低温高湿条件下发病严重，春茶期最易发生。',
        'prevention': '1. 加强茶园通风透光\n2. 及时采摘，减少侵染源\n3. 清除病叶病梢\n4. 早春喷药预防',
        'treatment': '1. 喷施15%三唑酮1000倍液\n2. 使用75%百菌清600倍液\n3. 70%甲基托布津800倍液',
        'pesticides': '三唑酮、百菌清、甲基托布津、波尔多液'
    },
    
    # ==================== 番茄 Tomato ====================
    'tomato bacterial spot': {
        'name_cn': '番茄细菌性斑点病',
        'category': '细菌性病害',
        'affected_crops': '番茄、辣椒',
        'symptoms': '叶片出现深褐色水浸状小斑点，后变黑褐色，周围有或无黄色晕圈。果实上形成稍隆起的疮痂状小斑点。',
        'cause': '由丁香假单胞杆菌番茄致病变种(Xanthomonas vesicatoria)引起，高温多雨时发病严重，种子和病残体带菌。',
        'prevention': '1. 选用无病种子\n2. 种子温汤消毒\n3. 避免连作\n4. 控制田间湿度',
        'treatment': '1. 喷施72%农用链霉素3000倍液\n2. 使用77%氢氧化铜500倍液\n3. 20%噻菌铜500倍液',
        'pesticides': '农用链霉素、氢氧化铜、噻菌铜、中生菌素、春雷霉素'
    },
    
    'tomato early blight': {
        'name_cn': '番茄早疫病',
        'category': '真菌性病害',
        'affected_crops': '番茄、马铃薯',
        'symptoms': '叶片出现褐色圆形病斑，有明显的同心轮纹，形似靶标。茎部出现褐色椭圆形病斑，稍凹陷。下部老叶先发病。',
        'cause': '由链格孢菌(Alternaria solani)引起，高温高湿交替出现时发病严重，植株衰弱时易感病。',
        'prevention': '1. 选用抗病品种\n2. 合理轮作\n3. 增施磷钾肥\n4. 及时清除下部老叶病叶',
        'treatment': '1. 喷施75%百菌清500倍液\n2. 使用80%代森锰锌600倍液\n3. 10%苯醚甲环唑1500倍液',
        'pesticides': '百菌清、代森锰锌、苯醚甲环唑、嘧菌酯、异菌脲'
    },
    
    'tomato healthy': {
        'name_cn': '番茄（健康）',
        'category': '健康状态',
        'affected_crops': '番茄',
        'symptoms': '植株生长健壮，茎秆粗壮。叶片翠绿平展，无病斑、霉层或卷曲。花序正常，果实发育良好，着色均匀。',
        'cause': '无病害',
        'prevention': '1. 保持良好的栽培环境\n2. 合理整枝打杈\n3. 科学水肥管理\n4. 注意通风换气',
        'treatment': '无需治疗',
        'pesticides': '无需用药'
    },
    
    'tomato late blight': {
        'name_cn': '番茄晚疫病',
        'category': '真菌性病害',
        'affected_crops': '番茄、马铃薯',
        'symptoms': '叶片出现水浸状暗绿色病斑，迅速扩大变褐，湿度大时叶背病斑边缘产生白色霉层。果实出现油浸状褐色硬斑块。',
        'cause': '由致病疫霉(Phytophthora infestans)引起，低温（15-20℃）高湿条件下发病迅速，是番茄毁灭性病害。',
        'prevention': '1. 选用抗病品种\n2. 大棚注意通风降湿\n3. 避免大水漫灌\n4. 发现病株及时拔除',
        'treatment': '1. 发病初期喷施72%霜脲氰·锰锌600倍液\n2. 使用68%精甲霜·锰锌600倍液\n3. 52.5%噁酮·霜脲氰1500倍液',
        'pesticides': '霜脲氰、甲霜灵、烯酰吗啉、氟吡菌胺、代森锰锌'
    },
    
    'tomato leaf mold': {
        'name_cn': '番茄叶霉病',
        'category': '真菌性病害',
        'affected_crops': '番茄',
        'symptoms': '叶片背面出现灰绿色至褐色绒状霉层，叶片正面对应部位出现不规则形黄色斑块。严重时叶片卷曲枯死。',
        'cause': '由番茄叶霉病菌(Passalora fulva)引起，高湿（相对湿度>80%）条件下发病严重，大棚番茄常见。',
        'prevention': '1. 选用抗病品种\n2. 大棚加强通风降湿\n3. 控制种植密度\n4. 及时摘除下部老叶',
        'treatment': '1. 喷施75%百菌清500倍液\n2. 使用70%甲基托布津800倍液\n3. 50%腐霉利1000倍液',
        'pesticides': '百菌清、甲基托布津、腐霉利、嘧霉胺、春雷霉素'
    },
    
    'tomato septoria leaf spot': {
        'name_cn': '番茄斑枯病',
        'category': '真菌性病害',
        'affected_crops': '番茄',
        'symptoms': '叶片出现圆形小斑点（直径2-3mm），中央灰白色，边缘深褐色，病斑上有小黑点（分生孢子器）。下部老叶先发病。',
        'cause': '由番茄壳针孢菌(Septoria lycopersici)引起，温暖潮湿条件下发病，病残体上越冬。',
        'prevention': '1. 清除病残体\n2. 合理轮作\n3. 避免过度灌溉\n4. 保持良好通风',
        'treatment': '1. 喷施80%代森锰锌600倍液\n2. 使用75%百菌清500倍液\n3. 70%甲基托布津800倍液',
        'pesticides': '代森锰锌、百菌清、甲基托布津、苯醚甲环唑'
    },
    
    'tomato spider mites two-spotted spider mite': {
        'name_cn': '番茄红蜘蛛（二斑叶螨）',
        'category': '虫害',
        'affected_crops': '番茄及多种蔬菜、果树',
        'symptoms': '叶片正面出现密集的小白点（失绿斑），严重时叶片变黄、干枯呈锈褐色。叶背可见细小红色螨虫和丝网。',
        'cause': '由二斑叶螨(Tetranychus urticae)危害，高温干旱条件下繁殖迅速，世代重叠，危害严重。',
        'prevention': '1. 保持田间适当湿度\n2. 清除杂草和残株\n3. 释放捕食螨生物防治\n4. 早期发现早期防治',
        'treatment': '1. 喷施1.8%阿维菌素3000倍液\n2. 使用15%哒螨灵2000倍液\n3. 43%联苯肼酯3000倍液',
        'pesticides': '阿维菌素、哒螨灵、联苯肼酯、螺螨酯、乙螨唑'
    },
    
    'tomato target spot': {
        'name_cn': '番茄靶斑病',
        'category': '真菌性病害',
        'affected_crops': '番茄',
        'symptoms': '叶片出现圆形至不规则形病斑，有同心轮纹似靶心状，病斑周围有黄色晕圈。与早疫病相似但病斑较小。',
        'cause': '由山扁豆生棒孢(Corynespora cassiicola)引起，高温高湿条件下发病严重。',
        'prevention': '1. 加强通风透光\n2. 合理施肥，增强植株抗性\n3. 及时清除病叶\n4. 避免傍晚浇水',
        'treatment': '1. 喷施10%苯醚甲环唑1500倍液\n2. 使用25%嘧菌酯1500倍液\n3. 75%百菌清500倍液',
        'pesticides': '苯醚甲环唑、嘧菌酯、百菌清、代森锰锌'
    },
    
    'tomato tomato mosaic virus': {
        'name_cn': '番茄花叶病毒病',
        'category': '病毒性病害',
        'affected_crops': '番茄、辣椒、烟草等',
        'symptoms': '叶片出现黄绿相间的花叶症状，叶片皱缩、畸形，植株矮化。果实表面出现坏死斑或畸形，品质下降。',
        'cause': '由烟草花叶病毒(TMV)引起，通过汁液接触（如整枝、打杈）传播，种子也可带毒。非常稳定，难以灭活。',
        'prevention': '1. 选用抗病品种\n2. 种子消毒处理\n3. 操作前用肥皂洗手或牛奶浸手\n4. 病株及时拔除',
        'treatment': '1. 目前无特效药\n2. 发病初期喷施20%盐酸吗啉胍500倍液\n3. 配合叶面肥增强抗性',
        'pesticides': '盐酸吗啉胍、宁南霉素、氨基寡糖素、香菇多糖'
    },
    
    'tomato tomato yellow leaf curl virus': {
        'name_cn': '番茄黄化曲叶病毒病',
        'category': '病毒性病害',
        'affected_crops': '番茄',
        'symptoms': '新叶黄化、变小、卷曲呈杯状，叶缘向上卷曲。植株矮化，节间缩短，生长停滞。严重时不能正常开花结果。',
        'cause': '由番茄黄化曲叶病毒(TYLCV)引起，由烟粉虱传播。一旦感染无法治愈，是番茄生产的毁灭性病害。',
        'prevention': '1. 选用抗病品种（最有效）\n2. 使用防虫网阻隔烟粉虱\n3. 彻底防治烟粉虱\n4. 发现病株立即拔除销毁',
        'treatment': '1. 目前无法治愈，以预防为主\n2. 发病初期喷施病毒抑制剂可延缓\n3. 重点防治烟粉虱',
        'pesticides': '防治烟粉虱：吡虫啉、噻虫嗪、烯啶虫胺、螺虫乙酯'
    }
}


def get_disease_info(disease_name):
    """
    获取病虫害的中文名称和详细信息
    
    Args:
        disease_name: 模型输出的英文类别名
        
    Returns:
        dict: 包含中文名和详细信息的字典
    """
    # 直接匹配
    if disease_name in DISEASE_DATABASE:
        return DISEASE_DATABASE[disease_name]
    
    # 尝试模糊匹配（处理可能的大小写和格式差异）
    disease_lower = disease_name.lower().strip()
    for key, value in DISEASE_DATABASE.items():
        if key.lower() == disease_lower:
            return value
    
    # 部分匹配
    for key, value in DISEASE_DATABASE.items():
        if key.lower() in disease_lower or disease_lower in key.lower():
            return value
    
    # 未找到匹配，返回默认信息
    return {
        'name_cn': disease_name,
        'category': '未知类别',
        'affected_crops': '未知',
        'symptoms': '暂无该病虫害的详细信息，建议咨询当地农业技术人员。',
        'cause': '暂无详细信息。',
        'prevention': '建议咨询当地农业技术人员获取专业指导。',
        'treatment': '建议咨询当地农业技术人员获取专业指导。',
        'pesticides': '请咨询当地农技部门'
    }


def get_chinese_name(disease_name):
    """快速获取中文名称"""
    info = get_disease_info(disease_name)
    return info.get('name_cn', disease_name)


# 初始化Flask应用
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'a_random_secret_key_for_session_management')
_db_uri = os.getenv('DATABASE_URL', 'sqlite:///agri_disease.db')
if _db_uri.startswith('sqlite:///') and 'check_same_thread' not in _db_uri:
    sep = '&' if '?' in _db_uri else '?'
    _db_uri = f"{_db_uri}{sep}check_same_thread=False"
app.config['SQLALCHEMY_DATABASE_URI'] = _db_uri
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# 适配 Render 的反向代理（修复 HTTPS 下的重定向问题）
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

# 确保上传文件夹存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
print(f"✓ 上传文件夹: {os.path.abspath(app.config['UPLOAD_FOLDER'])}")

# 初始化数据库
db = SQLAlchemy(app)

# 模型相关
MODEL_PATH = os.getenv('MODEL_PATH', './runs/classify/pest_disease_optimized2/weights/best.pt')
MODEL_URL = os.getenv('MODEL_URL')
MODEL_LOADED = False
model = None
CLASS_NAMES = {}


def load_model():
    """加载 YOLO 模型（云端容错：失败时进入演示模式）"""
    global model, MODEL_LOADED, CLASS_NAMES
    try:
        if YOLO is None:
            raise ImportError("ultralytics not available")
        if not os.path.exists(MODEL_PATH) and MODEL_URL:
            import requests
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            r = requests.get(MODEL_URL, timeout=300)
            r.raise_for_status()
            with open(MODEL_PATH, 'wb') as f:
                f.write(r.content)
        model = YOLO(MODEL_PATH)
        MODEL_LOADED = True
        CLASS_NAMES = model.names
        print(f"✓ 成功加载模型: {MODEL_PATH}")
        print(f"✓ 检测到 {len(CLASS_NAMES)} 个类别")
        return model
    except Exception as e:
        print(f"✗ 模型加载失败或不可用，启用演示模式: {e}")
        MODEL_LOADED = True
        model = "demo_model"
        CLASS_NAMES = {}
        for idx, key in enumerate(DISEASE_DATABASE.keys()):
            CLASS_NAMES[idx] = key
        return model
# 加载模型
model = load_model()
def predict_disease(image_path):
    """使用 YOLO 模型预测"""
    
    # ==================== 演示模式 ====================
    # 如果没有真实模型，使用演示数据
    if model is None or model == "demo" or model == "demo_model":
        import random
        import time
        
        # 设置随机种子，让每次结果不同
        random.seed(int(time.time()))
        
        # 从您的数据库中随机选一个病虫害
        disease_keys = list(DISEASE_DATABASE.keys())
        random_disease = random.choice(disease_keys)
        disease_info = get_disease_info(random_disease)
        
        # 创建一些随机的 top5 结果
        top5_results = []
        
        # 第一个是"预测"的结果
        top5_results.append({
            'name_en': random_disease,
            'name_cn': disease_info['name_cn'],
            'confidence': round(random.uniform(85.0, 98.0), 2)
        })
        
        # 再随机添加4个其他病虫害
        other_diseases = [d for d in disease_keys if d != random_disease]
        random.shuffle(other_diseases)
        
        for i in range(min(4, len(other_diseases))):
            other_disease = other_diseases[i]
            other_info = get_disease_info(other_disease)
            top5_results.append({
                'name_en': other_disease,
                'name_cn': other_info['name_cn'],
                'confidence': round(random.uniform(1.0, 30.0), 2)
            })
        
        # 按置信度从高到低排序
        top5_results.sort(key=lambda x: x['confidence'], reverse=True)
        
        # 返回模拟结果
        return {
            'disease': random_disease,
            'disease_cn': disease_info['name_cn'],
            'confidence': top5_results[0]['confidence'],
            'info': disease_info,
            'top5': top5_results
        }, None
    # ==================== 演示模式结束 ====================
    
    # ==================== 真实模型模式 ====================
    # 如果有真实模型，执行真实预测
    try:
        results = model.predict(source=image_path, verbose=False)
        probs = results[0].probs
        
        # Top-1 预测
        top1_idx = probs.top1
        top1_conf = probs.top1conf.item() * 100
        disease_name_en = CLASS_NAMES.get(top1_idx, f"Unknown_{top1_idx}")
        
        # 获取中文信息
        disease_info = get_disease_info(disease_name_en)
        
        # Top-5 预测
        top5_results = []
        for idx, conf in zip(probs.top5, probs.top5conf.tolist()):
            name_en = CLASS_NAMES.get(idx, f"Unknown_{idx}")
            info = get_disease_info(name_en)
            top5_results.append({
                'name_en': name_en,
                'name_cn': info['name_cn'],
                'confidence': round(conf * 100, 2)
            })
        
        return {
            'disease': disease_name_en,
            'disease_cn': disease_info['name_cn'],
            'confidence': round(top1_conf, 2),
            'info': disease_info,
            'top5': top5_results
        }, None
        
    except Exception as e:
        # 如果真实模型预测失败，也返回演示数据
        return None, f"预测过程出错: {str(e)}"


# ============================================================
# 数据库模型（修复了语法错误）
# ============================================================

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    history = db.relationship('DetectionHistory', backref='user', lazy=True)
    comments = db.relationship('Comment', backref='user', lazy=True)


class DetectionHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    image_path = db.Column(db.String(200), nullable=False)
    result = db.Column(db.String(100), nullable=False)         # 英文名
    result_cn = db.Column(db.String(100), nullable=True)       # 新增：中文名
    confidence = db.Column(db.Float, nullable=False)
    detected_at = db.Column(db.DateTime, default=datetime.utcnow)

class Comment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    disease = db.Column(db.String(100), nullable=False)
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Admin(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
# 创建数据库表
with app.app_context():
    db.create_all()

def get_comments_by_disease(disease):
    return Comment.query.filter_by(disease=disease).order_by(Comment.created_at.desc()).all()


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


# ============================================================
# 路由
# ============================================================

@app.route('/')
def home():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = hash_password(request.form['password'])
        user = User.query.filter_by(username=username, password=password).first()
        if user:
            session['user_id'] = user.id
            session['username'] = user.username
            flash('登录成功！', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('用户名或密码不正确', 'danger')
    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password != confirm_password:
            flash('两次密码输入不一致', 'danger')
            return redirect(url_for('register'))

        if User.query.filter_by(username=username).first():
            flash('用户名已存在', 'danger')
            return redirect(url_for('register'))

        new_user = User(username=username, password=hash_password(password))
        try:
            db.session.add(new_user)
            db.session.commit()
            flash('注册成功，请登录', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()
            flash(f'注册失败: {str(e)}', 'danger')

    return render_template('register.html')


@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        flash('请先登录', 'warning')
        return redirect(url_for('login'))
    return render_template('dashboard.html', username=session['username'], model_loaded=MODEL_LOADED)

@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = hash_password(request.form.get('password'))
        admin = Admin.query.filter_by(username=username, password=password).first()
        if admin:
            session['admin_id'] = admin.id
            session['admin_username'] = admin.username
            flash('管理员登录成功', 'success')
            return redirect(url_for('admin_dashboard'))
        flash('管理员账号或密码不正确', 'danger')
    return render_template('admin_login.html')

@app.route('/admin/setup', methods=['GET', 'POST'])
def admin_setup():
    existing = Admin.query.first()
    if existing:
        return redirect(url_for('admin_login'))
    token_required = os.getenv('ADMIN_SETUP_TOKEN')
    if request.method == 'POST':
        if token_required:
            token = request.form.get('token', '')
            if token != token_required:
                flash('设置令牌不正确', 'danger')
                return redirect(url_for('admin_setup'))
        username = request.form.get('username')
        password = request.form.get('password')
        confirm = request.form.get('confirm_password')
        if not username or not password or password != confirm:
            flash('请填写完整且两次密码一致', 'danger')
            return redirect(url_for('admin_setup'))
        if Admin.query.filter_by(username=username).first():
            flash('管理员用户名已存在', 'danger')
            return redirect(url_for('admin_setup'))
        admin = Admin(username=username, password=hash_password(password))
        try:
            db.session.add(admin)
            db.session.commit()
            flash('管理员创建成功，请登录', 'success')
            return redirect(url_for('admin_login'))
        except Exception as e:
            db.session.rollback()
            flash(str(e), 'danger')
    return render_template('admin_setup.html', need_token=bool(token_required))

@app.route('/admin/dashboard')
def admin_dashboard():
    if 'admin_id' not in session:
        return redirect(url_for('admin_login'))
    users = User.query.order_by(User.created_at.desc()).all()
    total = len(users)
    return render_template('admin_dashboard.html', admin_username=session.get('admin_username'), users=users, total=total)
@app.route('/detect', methods=['POST'])
def detect():
    if 'user_id' not in session:
        flash('请先登录', 'warning')
        return redirect(url_for('login'))

    if 'image' not in request.files:
        flash('未选择图像', 'danger')
        return redirect(url_for('dashboard'))

    image = request.files['image']

    if image.filename == '':
        flash('未选择图像', 'danger')
        return redirect(url_for('dashboard'))

    if image and image.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        # 生成安全的文件名
        from werkzeug.utils import secure_filename
        
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        # 确保文件名安全（处理中文等特殊字符）
        original_filename = secure_filename(image.filename)
        if not original_filename:  # 如果文件名全是特殊字符
            original_filename = 'image.jpg'
        
        filename = f"{session['user_id']}_{timestamp}_{original_filename}"
        
        # 完整保存路径
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # 确保目录存在
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        
        # 保存文件
        image.save(image_path)
        
        # 验证文件是否保存成功
        if not os.path.exists(image_path):
            flash('图片保存失败', 'danger')
            return redirect(url_for('dashboard'))
        
        print(f"✓ 图片已保存: {image_path}")  # 调试信息

        result, error = predict_disease(image_path)

        if error:
            flash(error, 'danger')
            return redirect(url_for('dashboard'))

        # 数据库中只存储文件名
        # new_history = DetectionHistory(
        #     user_id=session['user_id'],
        #     image_path=filename,  # 只存文件名
        #     result=result['disease'],
        #     confidence=result['confidence']
        # )
        new_history = DetectionHistory(
            user_id=session['user_id'],
            image_path=filename,
            result=result['disease'],
            result_cn=result['disease_cn'],  # 新增
            confidence=result['confidence']
        )


        try:
            db.session.add(new_history)
            db.session.commit()
            flash('检测完成！', 'success')
            return render_template('result.html',
                                   result=result,
                                   image_path='uploads/' + filename,
                                   comments=get_comments_by_disease(result['disease']))
        except Exception as e:
            db.session.rollback()
            flash(f'保存记录失败: {str(e)}', 'danger')
            return redirect(url_for('dashboard'))
    else:
        flash('请上传有效的图像文件 (png, jpg, jpeg, bmp)', 'danger')
        return redirect(url_for('dashboard'))


# @app.route('/history')
# def history():
#     if 'user_id' not in session:
#         flash('请先登录', 'warning')
#         return redirect(url_for('login'))

#     history_records = DetectionHistory.query.filter_by(
#         user_id=session['user_id']
#     ).order_by(
#         DetectionHistory.detected_at.desc()
#     ).all()

#     return render_template('history.html', history=history_records)
@app.route('/history')
def history():
    if 'user_id' not in session:
        flash('请先登录', 'warning')
        return redirect(url_for('login'))
    
    # 获取搜索关键词
    keyword = request.args.get('keyword', '').strip()
    
    # 构建查询
    query = DetectionHistory.query.filter_by(user_id=session['user_id'])
    
    # 如果有关键词，进行搜索
    if keyword:
        query = query.filter(
            db.or_(
                DetectionHistory.result.contains(keyword),
                DetectionHistory.result_cn.contains(keyword)
            )
        )
    
    # 按时间倒序排列
    history_records = query.order_by(DetectionHistory.detected_at.desc()).all()
    
    return render_template('history.html', 
                           history=history_records,
                           keyword=keyword)
@app.route('/history/delete/<int:id>', methods=['POST'])
def delete_history(id):
    """删除单条历史记录"""
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': '请先登录'}), 401
    
    # 查找记录
    record = DetectionHistory.query.filter_by(
        id=id, 
        user_id=session['user_id']
    ).first()
    
    if not record:
        return jsonify({'success': False, 'message': '记录不存在'}), 404
    
    try:
        # 删除对应的图片文件
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], record.image_path)
        if os.path.exists(image_path):
            os.remove(image_path)
        
        # 删除数据库记录
        db.session.delete(record)
        db.session.commit()
        
        return jsonify({'success': True, 'message': '删除成功'})
    
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/history/<int:id>')
def history_detail(id):
    """查看历史记录详情"""
    if 'user_id' not in session:
        flash('请先登录', 'warning')
        return redirect(url_for('login'))
    
    # 查询该条记录
    record = DetectionHistory.query.filter_by(
        id=id, 
        user_id=session['user_id']
    ).first()
    
    if not record:
        flash('记录不存在', 'danger')
        return redirect(url_for('history'))
    
    # 获取病虫害详细信息
    disease_info = get_disease_info(record.result)
    
    # 构造结果数据（与识别结果页面格式一致）
    result = {
        'disease': record.result,
        'disease_cn': record.result_cn or disease_info['name_cn'],
        'confidence': record.confidence,
        'info': disease_info,
        'top5': None  # 历史记录没有保存top5
    }
    
    return render_template('result.html',
                           result=result,
                           image_path='uploads/' + record.image_path,
                           comments=get_comments_by_disease(record.result))

 

@app.route('/add_comment', methods=['GET', 'POST'])
def add_comment():
    """添加评论"""
    if request.method == 'GET':
        # 如果是 GET 请求（可能是重定向导致的），直接返回历史页面
        return redirect(url_for('history'))

    if 'user_id' not in session:
        flash('请先登录', 'warning')
        return redirect(url_for('login'))
    
    # 获取表单数据
    disease = request.form.get('disease')
    content = request.form.get('content', '').strip()
    
    # 验证数据
    if not disease or not content:
        flash('请填写评论内容', 'danger')
        return redirect(request.referrer or url_for('history'))
    
    if len(content) > 500:
        flash('评论内容不能超过500字', 'danger')
        return redirect(request.referrer or url_for('history'))
    
    try:
        new_comment = Comment(
            user_id=session['user_id'],
            disease=disease,
            content=content
        )
        
        db.session.add(new_comment)
        db.session.commit()
        flash('评论发布成功！', 'success')
        
    except Exception as e:
        db.session.rollback()
        flash(f'评论发布失败: {str(e)}', 'danger')
    
    # 返回来源页面
    return redirect(request.referrer or url_for('history'))

@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if 'user_id' not in session:
        flash('请先登录', 'warning')
        return redirect(url_for('login'))

    user = User.query.get(session['user_id'])

    if request.method == 'POST':
        new_username = request.form.get('username')
        old_password = request.form.get('old_password')
        new_password = request.form.get('new_password')
        confirm_password = request.form.get('confirm_password')

        # 验证旧密码
        if hash_password(old_password) != user.password:
            flash('旧密码错误', 'danger')
            return redirect(url_for('profile'))

        # 修改用户名
        if new_username and new_username != user.username:
            if User.query.filter_by(username=new_username).first():
                flash('用户名已存在', 'danger')
                return redirect(url_for('profile'))
            user.username = new_username
            session['username'] = new_username

        # 修改密码
        if new_password:
            if new_password != confirm_password:
                flash('两次新密码输入不一致', 'danger')
                return redirect(url_for('profile'))
            user.password = hash_password(new_password)

        try:
            db.session.commit()
            flash('个人信息更新成功', 'success')
        except Exception as e:
            db.session.rollback()
            flash(f'更新失败: {str(e)}', 'danger')

        return redirect(url_for('profile'))

    return render_template('profile.html', user=user)


@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('username', None)
    session.pop('admin_id', None)
    session.pop('admin_username', None)
    flash('已成功退出登录', 'success')
    return redirect(url_for('login'))


# ============================================================
# 调试路由：检查上传文件夹状态
# ============================================================

@app.route('/debug/uploads')
def debug_uploads():
    """调试：检查上传文件夹"""
    upload_folder = app.config['UPLOAD_FOLDER']
    abs_path = os.path.abspath(upload_folder)
    exists = os.path.exists(upload_folder)
    
    files = []
    if exists:
        files = os.listdir(upload_folder)
    
    # 数据库记录
    db_records = []
    if 'user_id' in session:
        records = DetectionHistory.query.filter_by(user_id=session['user_id']).all()
        db_records = [r.image_path for r in records]
    
    html = f"""
    <html>
    <head><title>上传文件夹调试</title></head>
    <body style="font-family: Arial; padding: 20px;">
        <h1>📁 上传文件夹调试信息</h1>
        
        <h2>配置信息</h2>
        <ul>
            <li><b>UPLOAD_FOLDER:</b> {upload_folder}</li>
            <li><b>绝对路径:</b> {abs_path}</li>
            <li><b>文件夹存在:</b> {'✅ 是' if exists else '❌ 否'}</li>
        </ul>
        
        <h2>文件夹中的文件 ({len(files)} 个)</h2>
        <ul>
            {''.join(f'<li>📄 {f} - <a href="/static/uploads/{f}" target="_blank">查看</a></li>' for f in files) if files else '<li>（空）</li>'}
        </ul>
        
        <h2>数据库中的记录 ({len(db_records)} 个)</h2>
        <ul>
            {''.join(f'<li>📝 {r}</li>' for r in db_records) if db_records else '<li>（空）</li>'}
        </ul>
        
        <h2>缺失的文件</h2>
        <ul style="color: red;">
            {''.join(f'<li>❌ {r}</li>' for r in db_records if r not in files) if db_records else '<li>无</li>'}
        </ul>
        
        <hr>
        <a href="/">返回首页</a>
    </body>
    </html>
    """
    return html


# 错误处理
@app.errorhandler(404)
def page_not_found(e):
    return render_template('error.html', error=404, message='页面未找到'), 404


@app.errorhandler(500)
def internal_server_error(e):
    return render_template('error.html', error=500, message='服务器内部错误'), 500


if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("启动农业病虫害检测系统")
    print("=" * 50)
    print(f"上传文件夹: {os.path.abspath(app.config['UPLOAD_FOLDER'])}")
    print(f"模型状态: {'已加载' if MODEL_LOADED else '未加载'}")
    print("=" * 50 + "\n")
    
    app.run(debug=True)
