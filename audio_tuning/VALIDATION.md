# Одноразовая валидация автомобильного аудиотракта

Пользовательских режима остаётся два: `quick` и `full`. Все физические записи ниже
делаются через `audio_tuning.py quick`, по одному ESS на запуск. Внутренний
`validate_quick_series.py` только анализирует уже сохранённые результаты.

Каждый входной результат обязан пройти единый ESS-валидатор. Старая analysis schema,
отсутствующая clock correction, неполный active ESS, другая позиция микрофона или
channel делают серию недействительной.

## 0. Входной тракт и топология

До акустических серий в профиле заполняются input_signal_path, speaker_topology,
measurement_reference и crossover_policy. Если присутствует OEM-источник с
заводским EQ/TA, точная рекомендация DSP блокируется до измерения и подтверждения
de-EQ. У текущего прямого AUX OEM-ступени нет, поэтому статус
direct_aux_no_oem_stage считается валидным без дополнительной коррекции.

Текущая Aura использует только общую пару 1-2: мидбас и пищалка разделены пассивно.
Поэтому изменение активного полосного кроссовера и полярности отдельного излучателя
заблокировано. Статус сохраняется в metadata и отчётах.

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

## 4. Фаза и полярность

Фазовый тест обязателен только при независимых полосных каналах, loopback или иной
общей временной опоре и достоверной фазе. Тогда артефакт phase_alignment должен
содержать подтверждённое улучшение суммы в зоне раздела; ориентир — не менее 1 dB.

При текущих раздельных часах и пассивной общей паре статус равен
unsupported_by_current_measurement_path. Полярность и абсолютная задержка по
раздельным прогонам не заявляются и не блокируют амплитудную настройку EQ/TA.

## 5. Матрица DSP

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

Загрузчик дополнительно требует `accepted=true` у матрицы и каждого control,
полный набор регуляторов, одинаковую частотную сетку, конечные значения, допустимую
симметрию boost/cut, ссылки на baseline/plus/minus measurements и совпадение профиля,
устройств и sample rate. Отвергнутая или неполная матрица не используется.

## 6. Сброс валидации

Повторить соответствующие серии после смены микрофона, ориентации, Windows-
обработки, входного gain, источника, прошивки магнитолы, схемы подключения или
динамиков. Результаты разных трактов объединять запрещено.

По согласованному исключению bool-поля `validation` остаются ручными и не защищены
криптографической подписью. Они разрешают расчёт технического DSP-шага, но не могут
создать состояние `confirmed_preset` без full-сравнения и отдельного прослушивания.

## 7. Автоматические проверки

```powershell
.\.venv\Scripts\python.exe run_tests.py
```

Automated tests do not validate acoustic accuracy.
New in-car measurements are required.
