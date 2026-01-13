"""Tests for SCIP symbol ID format extraction.

SCIP (Source Code Intelligence Protocol) provides a standard format for
representing code symbols across different languages. Symbol IDs follow
the format: <scheme> <package> <descriptor>+

This module tests:
- SCIP symbol ID generation
- Package path resolution
- Symbol descriptor building (for functions, classes, methods)
- Cross-language consistency
- Symbol ID parsing and decomposition
"""

import pytest
from pathlib import Path
from typing import Optional

from openmemory.api.indexing.ast_parser import (
    Language,
    SymbolType,
    Symbol,
    create_parser,
)
from openmemory.api.indexing.scip_symbols import (
    # Core types
    SCIPScheme,
    SCIPDescriptor,
    SCIPSymbolID,
    # Builder
    SymbolIDBuilder,
    # Package resolution
    PackageResolver,
    PythonPackageResolver,
    TypeScriptPackageResolver,
    JavaPackageResolver,
    # Extractor
    SCIPSymbolExtractor,
    # Factory
    create_extractor,
    # Exceptions
    InvalidSymbolError,
)


# =============================================================================
# SCIP Scheme Tests
# =============================================================================


class TestSCIPScheme:
    """Tests for SCIP scheme enumeration."""

    def test_scheme_values(self):
        """Scheme enum has expected values."""
        assert SCIPScheme.LOCAL.value == "local"
        assert SCIPScheme.SCIP_PYTHON.value == "scip-python"
        assert SCIPScheme.SCIP_TYPESCRIPT.value == "scip-typescript"
        assert SCIPScheme.SCIP_JAVA.value == "scip-java"

    def test_scheme_for_language(self):
        """Get scheme for language."""
        assert SCIPScheme.for_language(Language.PYTHON) == SCIPScheme.SCIP_PYTHON
        assert SCIPScheme.for_language(Language.TYPESCRIPT) == SCIPScheme.SCIP_TYPESCRIPT
        assert SCIPScheme.for_language(Language.TSX) == SCIPScheme.SCIP_TYPESCRIPT
        assert SCIPScheme.for_language(Language.JAVA) == SCIPScheme.SCIP_JAVA
        assert SCIPScheme.for_language(Language.GO) == SCIPScheme.SCIP_GO


# =============================================================================
# SCIP Descriptor Tests
# =============================================================================


class TestSCIPDescriptor:
    """Tests for SCIP symbol descriptors."""

    def test_namespace_descriptor(self):
        """Create namespace descriptor."""
        desc = SCIPDescriptor.namespace("mypackage")
        assert desc.suffix == "/"
        assert str(desc) == "mypackage/"

    def test_type_descriptor(self):
        """Create type (class) descriptor."""
        desc = SCIPDescriptor.type_("MyClass")
        assert desc.suffix == "#"
        assert str(desc) == "MyClass#"

    def test_term_descriptor(self):
        """Create term (function/variable) descriptor."""
        desc = SCIPDescriptor.term("my_function")
        assert desc.suffix == "."
        assert str(desc) == "my_function."

    def test_method_descriptor(self):
        """Create method descriptor."""
        desc = SCIPDescriptor.method("my_method")
        assert desc.suffix == "."
        assert str(desc) == "my_method."

    def test_parameter_descriptor(self):
        """Create parameter descriptor."""
        desc = SCIPDescriptor.parameter("arg")
        assert desc.suffix == ")"
        assert str(desc) == "arg)"

    def test_type_parameter_descriptor(self):
        """Create type parameter descriptor."""
        desc = SCIPDescriptor.type_parameter("T")
        assert desc.suffix == "]"
        assert str(desc) == "T]"

    def test_meta_descriptor(self):
        """Create meta descriptor."""
        desc = SCIPDescriptor.meta("metadata")
        assert desc.suffix == ":"
        assert str(desc) == "metadata:"


# =============================================================================
# SCIP Symbol ID Tests
# =============================================================================


class TestSCIPSymbolID:
    """Tests for SCIP symbol ID format."""

    def test_create_symbol_id(self):
        """Create symbol ID from components."""
        symbol_id = SCIPSymbolID(
            scheme=SCIPScheme.SCIP_PYTHON,
            package="myproject",
            descriptors=[
                SCIPDescriptor.namespace("module"),
                SCIPDescriptor.type_("MyClass"),
                SCIPDescriptor.method("my_method"),
            ],
        )
        assert symbol_id.scheme == SCIPScheme.SCIP_PYTHON
        assert symbol_id.package == "myproject"
        assert len(symbol_id.descriptors) == 3

    def test_symbol_id_string(self):
        """Convert symbol ID to string."""
        symbol_id = SCIPSymbolID(
            scheme=SCIPScheme.SCIP_PYTHON,
            package="myproject",
            descriptors=[
                SCIPDescriptor.namespace("utils"),
                SCIPDescriptor.term("helper_func"),
            ],
        )
        expected = "scip-python myproject utils/helper_func."
        assert str(symbol_id) == expected

    def test_parse_symbol_id(self):
        """Parse symbol ID from string."""
        id_str = "scip-python myproject utils/helper_func."
        symbol_id = SCIPSymbolID.parse(id_str)

        assert symbol_id.scheme == SCIPScheme.SCIP_PYTHON
        assert symbol_id.package == "myproject"
        assert len(symbol_id.descriptors) == 2

    def test_symbol_id_equality(self):
        """Symbol IDs are equal when components match."""
        id1 = SCIPSymbolID(
            scheme=SCIPScheme.SCIP_PYTHON,
            package="pkg",
            descriptors=[SCIPDescriptor.term("func")],
        )
        id2 = SCIPSymbolID(
            scheme=SCIPScheme.SCIP_PYTHON,
            package="pkg",
            descriptors=[SCIPDescriptor.term("func")],
        )
        assert id1 == id2

    def test_symbol_id_hash(self):
        """Symbol ID is hashable."""
        symbol_id = SCIPSymbolID(
            scheme=SCIPScheme.SCIP_PYTHON,
            package="pkg",
            descriptors=[SCIPDescriptor.term("func")],
        )
        # Should not raise
        hash(symbol_id)
        assert symbol_id in {symbol_id}

    def test_local_symbol_id(self):
        """Create local symbol ID without package."""
        symbol_id = SCIPSymbolID(
            scheme=SCIPScheme.LOCAL,
            package="",
            descriptors=[SCIPDescriptor.term("local_var")],
        )
        assert str(symbol_id) == "local  local_var."


# =============================================================================
# Symbol ID Builder Tests
# =============================================================================


class TestSymbolIDBuilder:
    """Tests for fluent symbol ID builder."""

    def test_builder_python_function(self):
        """Build Python function symbol ID."""
        symbol_id = (
            SymbolIDBuilder()
            .scheme(SCIPScheme.SCIP_PYTHON)
            .package("myproject")
            .namespace("utils")
            .term("helper_func")
            .build()
        )
        assert str(symbol_id) == "scip-python myproject utils/helper_func."

    def test_builder_python_class_method(self):
        """Build Python class method symbol ID."""
        symbol_id = (
            SymbolIDBuilder()
            .scheme(SCIPScheme.SCIP_PYTHON)
            .package("myproject")
            .namespace("models")
            .type_("User")
            .method("get_name")
            .build()
        )
        assert str(symbol_id) == "scip-python myproject models/User#get_name."

    def test_builder_typescript_function(self):
        """Build TypeScript function symbol ID."""
        symbol_id = (
            SymbolIDBuilder()
            .scheme(SCIPScheme.SCIP_TYPESCRIPT)
            .package("@myorg/utils")
            .namespace("string-helpers")
            .term("formatName")
            .build()
        )
        assert str(symbol_id) == "scip-typescript @myorg/utils string-helpers/formatName."

    def test_builder_java_method(self):
        """Build Java method symbol ID."""
        symbol_id = (
            SymbolIDBuilder()
            .scheme(SCIPScheme.SCIP_JAVA)
            .package("com.example.myapp")
            .namespace("services")
            .type_("UserService")
            .method("findById")
            .build()
        )
        expected = "scip-java com.example.myapp services/UserService#findById."
        assert str(symbol_id) == expected

    def test_builder_nested_namespace(self):
        """Build symbol ID with nested namespaces."""
        symbol_id = (
            SymbolIDBuilder()
            .scheme(SCIPScheme.SCIP_PYTHON)
            .package("myproject")
            .namespace("pkg")
            .namespace("subpkg")
            .namespace("module")
            .term("func")
            .build()
        )
        assert str(symbol_id) == "scip-python myproject pkg/subpkg/module/func."

    def test_builder_meta_descriptor(self):
        """Build symbol ID with meta descriptor."""
        symbol_id = (
            SymbolIDBuilder()
            .scheme(SCIPScheme.SCIP_PYTHON)
            .package("myproject")
            .namespace("models")
            .type_("User")
            .meta("field")
            .term("email")
            .build()
        )
        assert str(symbol_id) == "scip-python myproject models/User#field:email."


# =============================================================================
# Package Resolver Tests
# =============================================================================


class TestPythonPackageResolver:
    """Tests for Python package resolution."""

    @pytest.fixture
    def resolver(self):
        """Create Python package resolver."""
        return PythonPackageResolver()

    def test_resolve_simple_path(self, resolver, tmp_path):
        """Resolve package from simple file path."""
        (tmp_path / "module.py").write_text("def func(): pass")

        package = resolver.resolve(tmp_path / "module.py", tmp_path)
        assert package == "module"

    def test_resolve_nested_path(self, resolver, tmp_path):
        """Resolve package from nested path."""
        pkg_dir = tmp_path / "pkg" / "subpkg"
        pkg_dir.mkdir(parents=True)
        (pkg_dir / "__init__.py").write_text("")
        (pkg_dir / "module.py").write_text("def func(): pass")

        package = resolver.resolve(pkg_dir / "module.py", tmp_path)
        assert package == "pkg.subpkg.module"

    def test_resolve_init_file(self, resolver, tmp_path):
        """Resolve package from __init__.py."""
        pkg_dir = tmp_path / "pkg"
        pkg_dir.mkdir()
        (pkg_dir / "__init__.py").write_text("def func(): pass")

        package = resolver.resolve(pkg_dir / "__init__.py", tmp_path)
        assert package == "pkg"


class TestTypeScriptPackageResolver:
    """Tests for TypeScript package resolution."""

    @pytest.fixture
    def resolver(self):
        """Create TypeScript package resolver."""
        return TypeScriptPackageResolver()

    def test_resolve_src_path(self, resolver, tmp_path):
        """Resolve package from src/ directory."""
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        (src_dir / "utils.ts").write_text("export function helper() {}")

        package = resolver.resolve(src_dir / "utils.ts", tmp_path)
        assert package == "src/utils"

    def test_resolve_with_package_json(self, resolver, tmp_path):
        """Resolve package name from package.json."""
        (tmp_path / "package.json").write_text('{"name": "@myorg/utils"}')
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        (src_dir / "index.ts").write_text("export function main() {}")

        package = resolver.resolve(src_dir / "index.ts", tmp_path)
        assert "@myorg/utils" in package


class TestJavaPackageResolver:
    """Tests for Java package resolution."""

    @pytest.fixture
    def resolver(self):
        """Create Java package resolver."""
        return JavaPackageResolver()

    def test_resolve_from_package_declaration(self, resolver, tmp_path):
        """Resolve package from package declaration in source."""
        java_file = tmp_path / "Main.java"
        java_file.write_text("""
package com.example.myapp;

public class Main {
    public static void main(String[] args) {}
}
""")
        package = resolver.resolve(java_file, tmp_path)
        assert package == "com.example.myapp"

    def test_resolve_default_package(self, resolver, tmp_path):
        """Resolve default package when no declaration."""
        java_file = tmp_path / "Main.java"
        java_file.write_text("""
public class Main {
    public static void main(String[] args) {}
}
""")
        package = resolver.resolve(java_file, tmp_path)
        assert package == ""  # Default package


# =============================================================================
# SCIP Symbol Extractor Tests
# =============================================================================


class TestSCIPSymbolExtractor:
    """Tests for SCIP symbol extraction from parsed symbols."""

    @pytest.fixture
    def extractor(self, tmp_path):
        """Create SCIP symbol extractor."""
        return create_extractor(tmp_path)

    def test_extract_python_function(self, extractor, tmp_path):
        """Extract SCIP symbol from Python function."""
        symbol = Symbol(
            name="helper_func",
            symbol_type=SymbolType.FUNCTION,
            line_start=1,
            line_end=3,
            language=Language.PYTHON,
        )

        file_path = tmp_path / "utils.py"
        scip_id = extractor.extract(symbol, file_path)

        assert scip_id.scheme == SCIPScheme.SCIP_PYTHON
        assert "utils" in str(scip_id)
        assert "helper_func" in str(scip_id)

    def test_extract_python_class(self, extractor, tmp_path):
        """Extract SCIP symbol from Python class."""
        symbol = Symbol(
            name="MyClass",
            symbol_type=SymbolType.CLASS,
            line_start=1,
            line_end=10,
            language=Language.PYTHON,
        )

        file_path = tmp_path / "models.py"
        scip_id = extractor.extract(symbol, file_path)

        assert "MyClass#" in str(scip_id)

    def test_extract_python_method(self, extractor, tmp_path):
        """Extract SCIP symbol from Python method."""
        symbol = Symbol(
            name="get_value",
            symbol_type=SymbolType.METHOD,
            line_start=5,
            line_end=7,
            language=Language.PYTHON,
            parent_name="MyClass",
        )

        file_path = tmp_path / "models.py"
        scip_id = extractor.extract(symbol, file_path)

        assert "MyClass#" in str(scip_id)
        assert "get_value." in str(scip_id)

    def test_extract_typescript_function(self, extractor, tmp_path):
        """Extract SCIP symbol from TypeScript function."""
        symbol = Symbol(
            name="formatName",
            symbol_type=SymbolType.FUNCTION,
            line_start=1,
            line_end=3,
            language=Language.TYPESCRIPT,
        )

        file_path = tmp_path / "utils.ts"
        scip_id = extractor.extract(symbol, file_path)

        assert scip_id.scheme == SCIPScheme.SCIP_TYPESCRIPT
        assert "formatName" in str(scip_id)

    def test_extract_typescript_interface(self, extractor, tmp_path):
        """Extract SCIP symbol from TypeScript interface."""
        symbol = Symbol(
            name="User",
            symbol_type=SymbolType.INTERFACE,
            line_start=1,
            line_end=5,
            language=Language.TYPESCRIPT,
        )

        file_path = tmp_path / "types.ts"
        scip_id = extractor.extract(symbol, file_path)

        assert "User#" in str(scip_id)

    def test_extract_typescript_field(self, extractor, tmp_path):
        """Extract SCIP symbol from TypeScript class field."""
        symbol = Symbol(
            name="email",
            symbol_type=SymbolType.FIELD,
            line_start=3,
            line_end=3,
            language=Language.TYPESCRIPT,
            parent_name="User",
        )

        file_path = tmp_path / "models.ts"
        scip_id = extractor.extract(symbol, file_path)

        assert "User#field:email." in str(scip_id)

    def test_extract_java_class(self, extractor, tmp_path):
        """Extract SCIP symbol from Java class."""
        symbol = Symbol(
            name="UserService",
            symbol_type=SymbolType.CLASS,
            line_start=3,
            line_end=20,
            language=Language.JAVA,
        )

        file_path = tmp_path / "UserService.java"
        scip_id = extractor.extract(symbol, file_path)

        assert scip_id.scheme == SCIPScheme.SCIP_JAVA
        assert "UserService#" in str(scip_id)

    def test_extract_nested_class(self, extractor, tmp_path):
        """Extract SCIP symbol from nested class."""
        symbol = Symbol(
            name="Inner",
            symbol_type=SymbolType.CLASS,
            line_start=5,
            line_end=10,
            language=Language.PYTHON,
            parent_name="Outer",
        )

        file_path = tmp_path / "nested.py"
        scip_id = extractor.extract(symbol, file_path)

        assert "Outer#" in str(scip_id)
        assert "Inner#" in str(scip_id)


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests with AST parser."""

    @pytest.fixture
    def parser(self):
        """Create AST parser."""
        return create_parser()

    @pytest.fixture
    def extractor(self, tmp_path):
        """Create SCIP extractor."""
        return create_extractor(tmp_path)

    def test_parse_and_extract_python(self, parser, extractor, tmp_path):
        """Parse Python file and extract SCIP symbols."""
        py_file = tmp_path / "example.py"
        py_file.write_text("""
class Calculator:
    def add(self, a: int, b: int) -> int:
        return a + b

    def subtract(self, a: int, b: int) -> int:
        return a - b

def standalone_func():
    pass
""")
        result = parser.parse_file(py_file)
        assert result.success

        scip_ids = [extractor.extract(s, py_file) for s in result.symbols]

        # Should have class, two methods, and standalone function
        assert len(scip_ids) >= 4

        # Check class symbol
        class_ids = [s for s in scip_ids if "Calculator#" in str(s) and "." not in str(s).split("#")[-1]]
        assert len(class_ids) >= 1

    def test_parse_and_extract_typescript(self, parser, extractor, tmp_path):
        """Parse TypeScript file and extract SCIP symbols."""
        ts_file = tmp_path / "example.ts"
        ts_file.write_text("""
interface User {
    name: string;
    age: number;
}

function createUser(name: string, age: number): User {
    return { name, age };
}

class UserManager {
    private users: User[] = [];

    addUser(user: User): void {
        this.users.push(user);
    }
}
""")
        result = parser.parse_file(ts_file)
        assert result.success

        scip_ids = [extractor.extract(s, ts_file) for s in result.symbols]

        # Check we got interface, function, and class
        id_strings = [str(s) for s in scip_ids]
        assert any("User#" in s for s in id_strings)
        assert any("createUser" in s for s in id_strings)
        assert any("UserManager#" in s for s in id_strings)

    def test_symbol_uniqueness(self, parser, extractor, tmp_path):
        """Each symbol should have a unique SCIP ID."""
        py_file = tmp_path / "unique.py"
        py_file.write_text("""
class A:
    def method(self): pass

class B:
    def method(self): pass
""")
        result = parser.parse_file(py_file)

        scip_ids = [extractor.extract(s, py_file) for s in result.symbols]
        id_strings = [str(s) for s in scip_ids]

        # All IDs should be unique
        assert len(id_strings) == len(set(id_strings))


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.fixture
    def extractor(self, tmp_path):
        """Create extractor."""
        return create_extractor(tmp_path)

    def test_special_characters_in_name(self, extractor, tmp_path):
        """Handle special characters in symbol names."""
        symbol = Symbol(
            name="__init__",
            symbol_type=SymbolType.METHOD,
            line_start=1,
            line_end=3,
            language=Language.PYTHON,
            parent_name="MyClass",
        )

        file_path = tmp_path / "test.py"
        scip_id = extractor.extract(symbol, file_path)

        assert "__init__" in str(scip_id)

    def test_unicode_in_name(self, extractor, tmp_path):
        """Handle unicode in symbol names."""
        symbol = Symbol(
            name="calculate_日本語",
            symbol_type=SymbolType.FUNCTION,
            line_start=1,
            line_end=3,
            language=Language.PYTHON,
        )

        file_path = tmp_path / "test.py"
        scip_id = extractor.extract(symbol, file_path)

        assert "calculate_日本語" in str(scip_id)

    def test_deeply_nested_path(self, extractor, tmp_path):
        """Handle deeply nested file paths."""
        deep_dir = tmp_path / "a" / "b" / "c" / "d" / "e"
        deep_dir.mkdir(parents=True)

        symbol = Symbol(
            name="deep_func",
            symbol_type=SymbolType.FUNCTION,
            line_start=1,
            line_end=3,
            language=Language.PYTHON,
        )

        file_path = deep_dir / "module.py"
        scip_id = extractor.extract(symbol, file_path)

        # Should contain all path components
        assert "deep_func" in str(scip_id)

    def test_import_symbol(self, extractor, tmp_path):
        """Handle import symbols."""
        symbol = Symbol(
            name="os",
            symbol_type=SymbolType.IMPORT,
            line_start=1,
            line_end=1,
            language=Language.PYTHON,
        )

        file_path = tmp_path / "test.py"
        scip_id = extractor.extract(symbol, file_path)

        # Imports should still get valid IDs
        assert scip_id is not None
        assert "os" in str(scip_id)
