"""Publish short forecasts on the requested day range, including all decisions."""
from copy import deepcopy


def select_daily_report(result: dict, requested_days: list[int]) -> dict:
    days = sorted(set(int(day) for day in requested_days))
    if not days or days != list(range(1, days[-1] + 1)) or days[-1] > 90:
        raise ValueError('短时预测必须连续选择第1天至第N天（最多90天）')
    rows = result['short_term_forecast']
    if len(rows) != len(days):
        raise ValueError('短时预测结果与选择天数不一致')
    target = rows[-1]
    current = result['current_assessment']
    snapshot = result['input_snapshot']
    standard = result['applied_standard']
    result.update(forecast_scope='short', forecast_days=days[-1],
                  selected_target_prediction=target, long_term_forecast=[])
    result['future_weather_note'] = f'本次逐日评估未来{days[-1]}天；已填天气按对应日期计算，未填项目沿用观测情景。'
    result['calendar_context'].update(
        selected_target_date=target['target_date'], selected_horizon_days=days[-1],
        selected_target_year_month=target['target_date'])
    trend = result['growth_trend']
    for key, predicted, original in [('height_change','predicted_height','height'),
                                     ('diameter_change','predicted_diameter','diameter'),
                                     ('crown_width_change','predicted_crown_width','crown_width'),
                                     ('clearance_change','predicted_clearance_distance','clearance_distance')]:
        trend[key] = (round(target[predicted] - snapshot[original], 3)
                      if target.get(predicted) is not None and snapshot.get(original) is not None else None)
    trend.update(forecast_days=days[-1], direction='按逐日天气与树木生长情景计算')
    timing = deepcopy(result.get('threshold_timing') or {})
    timing['scan_days'] = days[-1]
    for value in timing.values():
        if isinstance(value, dict) and value.get('days') is not None and value['days'] > days[-1]:
            value.update(days=None, target_date=None, within_scan_window=False, reason=f'本次未来{days[-1]}天内未达到')
    result['threshold_timing'] = timing
    all_rows = [dict(current, target_date=result['calendar_context']['observation_date']), *rows]
    peak = max(rows, key=lambda item: item['comprehensive_risk_index'])
    minimum = result['risk_timeline_analysis']['minimum_clearance_standard']
    first = next((row for row in all_rows if
                  row['comprehensive_risk_index'] >= standard['pruning_trigger_hazard_index']
                  or row.get('predicted_effective_clearance', row.get('standard_analysis', {}).get('effective_clearance', snapshot['clearance_distance'])) < minimum
                  or row.get('contact_risk', {}).get('requires_immediate_pruning')), None)
    status = '立即处置' if first is all_rows[0] else '计划修剪' if first else '继续监测'
    deadline = (f"{first['target_date']}前复测并安排处置" if first else f"本次未来{days[-1]}天内未达到处置阈值，保持巡检")
    maintenance = dict(status=status, requires_trimming=bool(first), recommended_deadline=deadline,
                       recommended_trim_date=first['target_date'] if first else None,
                       recommended_trim_month=None, trigger_month=None,
                       trigger_reason='逐日风险指数、树线关系与有效净空联合判断',
                       days_to_trigger=all_rows.index(first) if first else None)
    timeline = [dict(row, day_index=index+1, target_label=f"第{index+1}天（{row['target_date']}）") for index,row in enumerate(rows)]
    result['risk_timeline_analysis'] = dict(
        minimum_clearance_standard=minimum, threshold_timing=timing, timeline=timeline,
        peak_risk=dict(peak, target_year_month=peak['target_date']),
        first_significant_increase=None, risk_increase_conclusion=f"本次未来{days[-1]}天风险峰值日期为{peak['target_date']}",
        maintenance_decision=maintenance,
        planning_risk=dict(risk_level='高风险' if first else '低风险', decision=status,
                           peak_date=peak['target_date'], conclusion=deadline))
    result['treatment_advice'].update(action=status, deadline=deadline,
        trigger=maintenance['trigger_reason'], target_month_goal=f'保持有效净空不低于{minimum}m')
    result['requested_target_date'] = target['target_date']
    from algorithms.prediction.risk_evaluation import calculation_ledger
    from algorithms.prediction.standards import PredictionStandards
    result['calculation_ledger'] = calculation_ledger(snapshot, current, target,
        target['comprehensive_risk'], PredictionStandards.from_mapping(standard))
    result['work_order']['urgency'] = max(all_rows, key=lambda item:item['comprehensive_risk_index'])['hazard_level']
    result['work_order']['comprehensive_urgency'] = result['work_order']['urgency']
    result['work_order']['urgency_icon'] = ''
    return result
