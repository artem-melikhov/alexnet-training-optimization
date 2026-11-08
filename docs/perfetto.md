# Как посмотреть трейс в Perfetto

В Perfetto нажмите **Open trace file** и выберите JSON. Раскройте группу процесса Python и дорожку CUDA. Найдите `ProfilerStep#4` на дорожке CPU. Вложенные блоки показывают, где Python запускает операции и где ждёт. На дорожке CUDA видно, когда GPU выполняет поставленную в очередь работу.

Чтобы получить среднее, минимальное и максимальное время пяти шагов, откройте **Query (SQL)** и выполните запрос.

```sql
SELECT COUNT(*) AS steps,
       ROUND(AVG(dur) / 1e6, 2) AS avg_ms,
       ROUND(MIN(dur) / 1e6, 2) AS min_ms,
       ROUND(MAX(dur) / 1e6, 2) AS max_ms
FROM slice
WHERE name GLOB 'ProfilerStep#[4-8]'
  AND category = 'user_annotation' AND dur > 0;
```

Время копирования можно посмотреть у события `Memcpy HtoD` на дорожке GPU. Блок `02_to_device` на CPU может быть гораздо длиннее, потому что вызов ждёт завершения операций, которые уже стоят в очереди.
