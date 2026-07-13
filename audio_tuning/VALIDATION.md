# Одноразовая валидация автомобильного аудиотракта

Пользовательских режима остаётся два: `quick` и `full`. Все физические записи ниже
делаются через `audio_tuning.py quick`, по одному ESS на запуск. Внутренний
`validate_quick_series.py` только анализирует уже сохранённые результаты.

## 1. Повторяемость

Не меняя микрофон, громкость, preset и тракт, выполнить три quick-замера. Создать
manifest:

```json
{
  "schema_version": "quick_validation_manifest_v1",
  "kind": "repeatability",
  "device_profile": "../devices/aura_indigo_877dsp_mkii.json",
  "results": ["repeat_1/ess_response.json", "repeat_2/ess_response.json", "repeat_3/ess_response.json"]
}
```

```powershell
.\.venv\Scripts\python.exe validate_quick_series.py .\manifests\repeatability.json `
  --out .\data\validation\aura_repeatability.json
```

Для `100-10000 Hz` стандартное отклонение каждой полосы должно быть не выше лимита
профиля, по умолчанию 1 dB. Принятый файл указывается в
`validation_artifacts.repeatability`, затем ставится
`validation.repeatability_verified=true`.

## 2. Линейность по громкости

Выполнить quick для тихого, рабочего и громкого уровней с неизменным preset. В
каждом прогоне профиль должен содержать фактические source/head-unit volume.

```json
{
  "schema_version": "quick_validation_manifest_v1",
  "kind": "level_linearity",
  "device_profile": "../devices/aura_indigo_877dsp_mkii.json",
  "measurements": [
    {"level_rank": 1, "result": "quiet/ess_response.json"},
    {"level_rank": 2, "result": "reference/ess_response.json"},
    {"level_rank": 3, "result": "loud/ess_response.json"}
  ]
}
```

Проверяются монотонный рост относительного уровня и совпадение формы после robust
alignment с допуском 1 dB. Принятый файл указывается в
`validation_artifacts.level_linearity`, затем ставится
`validation.volume_linearity_verified=true`.

## 3. Обработка микрофона

В Windows вручную отключить AGC, AEC, noise suppression и enhancements. Результат
не выводится автоматически из уровневого теста: отсутствие изменения формы снижает
риск, но не доказывает отключение обработки. После проверки поставить
`validation.microphone_processing_disabled=true`.

## 4. Матрица DSP

После заполнения всех регуляторов выполнить baseline и для каждого control два
quick-замера: `+Δ` и `−Δ`, остальные настройки неизменны. Manifest содержит
`baseline_result` и для каждого control поля `id`, `plus_delta_db`, `plus_result`,
`minus_delta_db`, `minus_result`. Валидатор вычисляет среднее воздействие на 1 dB и
ошибку симметрии boost/cut.

Матрица строится по 512-точечной 1/6-октавной кривой ESS, а не по 31 сводной
полосе. Это обязательно для 48-полосного параметрического EQ Aura.

Принятый результат подключается как
`dsp_control_model.response_matrix_file`; состояние меняется на `characterized`, а
`validation.dsp_controls_characterized` на `true`. До этого full строит отчёт, но
не выдаёт автоматическую рекомендацию.

## 5. Сброс валидации

Повторить соответствующие серии после смены микрофона, ориентации, Windows-
обработки, входного gain, источника, прошивки магнитолы, схемы подключения или
динамиков. Результаты разных трактов объединять запрещено.
