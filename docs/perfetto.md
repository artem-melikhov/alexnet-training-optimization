
# Чтение трейса в Perfetto

Откройте JSON через Open trace file. Раскройте группу процесса Python и CUDA stream.
На CPU найдите `ProfilerStep#4`. Отметки этапов показывают, где Python ставит работу
в очередь и где ждёт. CUDA-дорожка показывает фактическое выполнение ядер.

Длительность пяти записанных шагов можно получить в Query (SQL).

```sql
SELECT COUNT(*) AS steps,
       ROUND(AVG(dur) / 1e6, 2) AS avg_ms,
       ROUND(MIN(dur) / 1e6, 2) AS min_ms,
       ROUND(MAX(dur) / 1e6, 2) AS max_ms
FROM slice
WHERE name GLOB 'ProfilerStep#[4-8]'
  AND category = 'user_annotation' AND dur > 0;
```

Для фактического копирования смотрите события `Memcpy HtoD` на GPU-дорожке.
CPU-время `02_to_device` может включать ожидание ранее поставленной работы.
