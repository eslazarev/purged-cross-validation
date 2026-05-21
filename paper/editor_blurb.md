# purgedcv — a short explainer for editors

*This page is written for a website editor who is about to link to the
package. It explains, in plain language, what the tool is and why it is
worth a link. A ready-to-paste blurb is at the bottom.*

## In one sentence

purgedcv is a free, open-source Python tool that helps data scientists
test their prediction models honestly.

## What it is

purgedcv is a small Python library for people who build prediction
models — forecasting electricity demand, prices, rainfall, equipment
failure, and similar tasks. It plugs into scikit-learn, the standard
machine-learning toolkit in Python, so it fits into work people already
do.

## The problem it solves

Before anyone trusts a prediction model, they test it. You hide part of
the data, ask the model to predict it, and check how close it got.

That test is easy to get wrong. With time-based data, the model can
accidentally catch a glimpse of the answers it is supposed to be
guessing. When that happens the test reports a great score, and the
model looks far smarter than it really is. Teams have shipped models —
and researchers have published results — on scores that were quietly
inflated this way.

purgedcv fixes the test, not the model. It removes the overlapping data
that leaks the answer, so the score you see is the score the model would
really earn in the real world. Sometimes the honest score is
disappointing. That is the point: better to find out now than after the
model is in production.

## Who it is for

Data scientists, machine-learning engineers, quantitative analysts, and
researchers — anyone working in Python who needs to know how accurate a
forecasting model really is.

## Why it is safe to link to

- Free and open source, MIT licensed — no cost, no sign-up.
- Published on PyPI; it installs in one line with `pip install purgedcv`.
- Documented, with worked examples on real public data sets.
- Fully type-checked and covered by an automated test suite.
- Built by an independent researcher; the underlying methods come from
  established, peer-reviewed literature, not invented for the package.

## Facts box

| | |
|---|---|
| Name | purgedcv |
| Install | `pip install purgedcv` |
| Code | https://github.com/eslazarev/purged-cross-validation |
| Documentation | https://eslazarev.github.io/purged-cross-validation/ |
| License | MIT (free for any use) |
| Language | Python 3.10 or newer |
| Author | Evgenii Lazarev, independent researcher |

## Ready-to-paste blurb (short)

> **purgedcv** is a free, open-source Python library that helps data
> scientists check their forecasting models honestly. Ordinary model
> tests can let a model peek at the answers it is meant to predict, which
> inflates its score; purgedcv removes that leak so the reported accuracy
> is the real one. It works with scikit-learn and installs with
> `pip install purgedcv`.

## Ready-to-paste blurb (one line)

> **purgedcv** — an open-source Python tool that keeps model-accuracy
> tests honest for forecasting and time-series data.

---

# Русская версия

*Простое пояснение и готовые блёрбы на русском языке.*

## Блёрб для вставки (подробный)

> **purgedcv** — это бесплатная библиотека для языка программирования
> Python. Она помогает честно проверять, насколько хорошо работают
> модели, которые что-то предсказывают: спрос на электричество на завтра,
> цены, погоду, момент поломки оборудования и тому подобное.
>
> **Зачем это нужно.** Прежде чем доверять такой модели, её проверяют.
> Прячут часть данных, просят модель угадать их и сравнивают догадку с
> настоящим ответом. Звучит просто — но тут легко ошибиться. Когда данные
> идут во времени, модель во время проверки может случайно «подсмотреть»
> подсказку к ответу. Тогда проверка показывает отличный результат, но
> это обман: на самом деле модель работает хуже, и в реальной работе это
> рано или поздно вскроется.
>
> Такая ошибка встречается очень часто. Из-за неё запускали продукты и
> даже публиковали научные статьи с завышенными, по сути ненастоящими
> цифрами точности.
>
> **Что делает purgedcv.** Он чинит саму проверку, а не модель. Он
> убирает те данные, которые дают подсказку, — и тогда цифра точности
> становится настоящей. Иногда честная цифра оказывается хуже, чем
> хотелось бы. В этом и польза: лучше узнать правду сейчас, чем уже после
> запуска.
>
> **Для кого.** Для всех, кто создаёт такие предсказывающие модели на
> Python — аналитиков, инженеров, исследователей. purgedcv работает
> вместе со scikit-learn, самым популярным набором инструментов для
> машинного обучения в Python, и встраивается в привычную работу.
>
> purgedcv бесплатный, с открытым исходным кодом, и устанавливается одной
> строкой: `pip install purgedcv`.

## Блёрб в одну фразу

> **purgedcv** — бесплатный инструмент для Python, который помогает честно
> измерять точность предсказывающих моделей и не обманываться завышенными
> цифрами.

## Карточка фактов

| | |
|---|---|
| Название | purgedcv |
| Установка | `pip install purgedcv` |
| Код | https://github.com/eslazarev/purged-cross-validation |
| Документация | https://eslazarev.github.io/purged-cross-validation/ |
| Лицензия | MIT (бесплатно для любого использования) |
| Язык | Python 3.10 и новее |
| Автор | Евгений Лазарев, независимый исследователь |
