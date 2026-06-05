from __future__ import annotations
from adapters.base import BaseAdapter

class FlaskAdapter(BaseAdapter):

    def _setup(self):
        self._app = None

    def start(self, **kwargs):
        self._engine.run_file(self._tri_file)
        self._app = self._engine.get('app')
        if self._app is None:
            raise RuntimeError(
                'flask adapter: no app variable found in .tri file. '
                'the .tri file must define a flask app, e.g. '
                'let app = flask.Flask("name")'
            )
        return self._app

    def test_client(self):
        if self._app is None:
            self.start()
        return self._app.test_client()

    def run(self, host='127.0.0.1', port=5000, **kwargs):
        app = self.start()
        app.run(host=host, port=port, **kwargs)
