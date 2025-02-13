-- Software Usage
SELECT sg.guid, sw.event_name AS sw_event_name, sw.frgnd_proc_dt AS dt, sw.frgnd_proc_name, sw.frgnd_proc_duration_ms 
FROM public.sample_table sg, university_prod.frgnd_backgrnd_apps_v4_hist sw 
WHERE sg.guid = sw.guid;

-- Web Usage
SELECT sg.guid, wb.dt, wb.browser, wb.parent_category AS web_parent_category, wb.sub_category AS web_sub_category, wb.duration_ms
FROM public.sample_table sg, university_prod.web_cat_usage_v2 wb
WHERE sg.guid = wb.guid;

-- Temperature
WITH pwr AS
(select guid, dt, event_name AS temp_event_name, duration_ms, metric_name, attribute_metric_level1 AS temp_attribute_metric_level1, nrs, avg_val
from university_prod.power_acdc_usage_v4_hist
where metric_name LIKE '%TEMPERATURE%')
select pwr.*
from pwr, public.sample_table sg
where pwr.guid = sg.guid;

-- CPU Usage (C_state table)
WITH samp AS (
    SELECT * FROM public.sample_table
),
core_count AS (
    SELECT c.guid, CAST(MAX(cpu_id) AS int) + 1 AS cores
    FROM university_prod.os_c_state c, samp
    WHERE c.guid = samp.guid
    AND c.cpu_id != '_TOTAL'
    GROUP BY c.guid
),
cpu_usage AS (
    SELECT c.guid, dt, SUM(sample_count * average) / SUM(sample_count) AS usage, SUM(sample_count) AS nrs
    FROM university_prod.os_c_state c, samp
    WHERE c.guid = samp.guid
    AND c.cpu_id = '_TOTAL'
    GROUP BY c.guid, dt
)
-- SELECT cu.guid, cu.dt, cc.cores, cu.usage, cc.cores * cu.usage AS norm_usage
SELECT cu.guid, cu.dt, cc.cores * cu.usage AS norm_usage, nrs
FROM core_count cc, cpu_usage cu
where cc.guid = cu.guid;

-- Target Variable, Power Usage
SELECT pw.guid, pw.dt, SUM(pw.nrs * pw.mean) / SUM(pw.nrs) AS mean, SUM(pw.nrs) AS nrs_sum
FROM public.sample_table sg, university_prod.hw_pack_run_avg_pwr pw
WHERE sg.guid = pw.guid
GROUP BY pw.guid, pw.dt;