Инструкция по запуску веб-приложения:
1. Установить Python 3.8+ (рекомендуемая версия 3.10.11); при установке проверить чекбокс "ADD to PATH", должен быть
включен. https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe - 64 бит Windows
2. Проверить правильно ли установлен Python, для этого введите в командной строке команду python --version
2. Разархивировать .zip архив в любую папку 'xxx'.
3. В командной строке ввести команду pip install -r <path_to_directory xxx>/xxx/requirements.txt
4. По окончанию установки библиотек необходимо запустить скрипт командой python <path_to_directory xxx>/xxx/main.py
5. Откроется ваш браузер по-умолчанию и запуститься веб-приложение
6. !!!Для преобразования моделей в PMML формат необходимо установить Java Runtime Environment (JRE), https://java-runtime.ru/download