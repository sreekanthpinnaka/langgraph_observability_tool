# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['D:\\observe\\langgraph_observe\\__main__.py'],
    pathex=[],
    binaries=[],
    datas=[('D:\\observe\\langgraph_observe\\server\\ui\\index.html', 'langgraph_observe/server/ui')],
    hiddenimports=['uvicorn', 'uvicorn.logging', 'uvicorn.loops', 'uvicorn.loops.auto', 'uvicorn.protocols', 'uvicorn.protocols.http', 'uvicorn.protocols.http.auto', 'uvicorn.protocols.http.h11_impl', 'uvicorn.protocols.http.httptools_impl', 'uvicorn.protocols.websockets', 'uvicorn.protocols.websockets.auto', 'uvicorn.lifespan', 'uvicorn.lifespan.on', 'uvicorn.lifespan.off', 'starlette', 'fastapi', 'pydantic', 'sqlalchemy', 'sqlalchemy.dialects.sqlite', 'sqlalchemy.dialects.mysql', 'sqlalchemy.sql.default_comparator', 'pymysql', 'dotenv', 'anyio'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='langgraph-observe',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
