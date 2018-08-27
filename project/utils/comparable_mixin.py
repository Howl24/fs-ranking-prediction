"""
Compara dos objetos a partir del valor que genere el método _cmpkey()

Uso:

    # Hacer que la clase a comparar herede de ComparableMixin
    class Foo(ComparableMixin):
        ...

        # Implementar un método _cmpkey() que retorne el elemento a usar para comparar.
        def _cmpkey():
            return self.foo_value
"""


class ComparableMixin(object):
    def _compare(self, other, method):
        try:
            return method(self._cmpkey(), other._cmpkey())
        except (AttributeError, TypeError):
            raise NotImplementedError("_cmpkey not implemented")

    def __lt__(self, other):
        return self._compare(other, lambda s,o: s < o)

    def __le__(self, other):
        return self._compare(other, lambda s,o: s <= o)

    def __eq__(self, other):
        return self._compare(other, lambda s,o: s == o)

    def __ge__(self, other):
        return self._compare(other, lambda s,o: s >= o)

    def __gt__(self, other):
        return self._compare(other, lambda s,o: s > o)

    def __ne__(self, other):
        return self._compare(other, lambda s,o: s != o)

    def __hash__(self):
        try:
            return hash(self._cmpkey())
        except (AttributeError, TypeError):
            raise NotImplementedError("_cmpkey not implemented")
