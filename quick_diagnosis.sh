#!/bin/bash
# quick_diagnosis.sh - Quick diagnosis of SOC Platform data flow issues

echo "🔍 SOC Platform Quick Diagnosis"
echo "================================"

# Check if containers are running
echo "🐳 Container Status:"
docker-compose ps

echo ""
echo "📊 ClickHouse Data Check:"
# Check ClickHouse data
docker exec soc-clickhouse clickhouse-client --query "
SELECT 
    'raw_logs' as table_name,
    count() as total_rows,
    countIf(timestamp >= now() - INTERVAL 1 HOUR) as recent_rows_1h,
    countIf(timestamp >= now() - INTERVAL 5 MINUTE) as recent_rows_5m
FROM raw_logs
UNION ALL
SELECT 
    'anomaly_scores' as table_name,
    count() as total_rows,
    countIf(timestamp >= now() - INTERVAL 1 HOUR) as recent_rows_1h,
    countIf(timestamp >= now() - INTERVAL 5 MINUTE) as recent_rows_5m
FROM anomaly_scores
UNION ALL
SELECT 
    'alerts' as table_name,
    count() as total_rows,
    countIf(timestamp >= now() - INTERVAL 1 HOUR) as recent_rows_1h,
    countIf(timestamp >= now() - INTERVAL 5 MINUTE) as recent_rows_5m
FROM alerts;
"

echo ""
echo "📋 Recent ML Pipeline Logs:"
docker logs --tail 10 soc-ml-pipeline | grep -E "(✅|❌|💾|📤|Error|Failed)"

echo ""
echo "📋 Recent Scoring Engine Logs:"
docker logs --tail 10 soc-scoring-engine | grep -E "(✅|❌|🚨|📥|Error|Failed)"

echo ""
echo "📋 Recent Alerting Logs:"
docker logs --tail 10 soc-alerting | grep -E "(✅|❌|📨|🚨|Error|Failed)"

echo ""
echo "🔧 DIAGNOSIS SUMMARY:"
echo "If you see:"
echo "  ✅ Only raw_logs have data → ML Pipeline not storing anomaly scores"
echo "  ✅ No anomaly_scores → Replace ML Pipeline with fixed version"
echo "  ✅ Scores exist but no alerts → Check Scoring Engine Kafka consumption"
echo "  ✅ Import errors → Services trying to import each other (microservices violation)"

echo ""
echo "💡 QUICK FIXES:"
echo "1. Replace run_ml_pipeline.py with run_ml_pipeline_fixed.py"
echo "2. Replace run_scoring_engine.py with run_scoring_engine_fixed.py"
echo "3. Restart containers: docker-compose restart ml-pipeline scoring-engine"