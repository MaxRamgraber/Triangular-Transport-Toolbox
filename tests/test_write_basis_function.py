import numpy as np
import pytest
from triangular_transport_toolbox import transport_map


@pytest.fixture
def simple_map():
    """Create a simple transport map for testing."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 2))
    # Simple map structure: constant + linear terms
    monotone = [[[0]], [[1]]]
    nonmonotone = [[[]], [[0]]]
    
    tm = transport_map(
        X=X,
        monotone=monotone,
        nonmonotone=nonmonotone,
        verbose=False
    )
    return tm


class TestWriteBasisFunctionConstant:
    """Tests for constant terms (empty list)."""

    def test_constant_standard_mode(self, simple_map):
        """Empty list should return ones for standard mode."""
        string, modifier_log = simple_map.write_basis_function(term=[], mode='standard')
        
        assert string == "np.ones(__x__.shape[:-1])"
        assert modifier_log == {"constant": None}

    def test_constant_derivative_mode(self, simple_map):
        """Empty list should return zeros for derivative mode."""
        string, modifier_log = simple_map.write_basis_function(term=[], mode='derivative', k=0)
        
        assert string == "np.zeros(__x__.shape[:-1])"
        assert modifier_log == {"constant": None}


class TestWriteBasisFunctionPolynomial:
    """Tests for polynomial terms (list of indices)."""

    def test_univariate_polynomial_standard(self, simple_map):
        """Single variable polynomial should return valid string.
        
        For polynomial terms, write_basis_function returns:
        - string: A key like "P_0_O_1" (Polynomial in variable 0, Order 1)
        - modifier_log: A dict with "variables" containing the actual expressions
        
        The key serves as a placeholder for common subexpression elimination.
        The actual NumPy polynomial expression is stored in modifier_log["variables"]
        and gets substituted later by function_constructor_alternative.
        
        Example modifier_log structure:
            {
                "variables": {
                    "P_0_O_1": "np.polynomial.HermiteE([0.0,1.0])(__x__[...,0])"
                }
            }
        
        Where [0.0, 1.0] are coefficients: 0.0*He_0(x) + 1.0*He_1(x) = x (linear)
        """
        # Term [0] means first-order polynomial in variable 0
        string, modifier_log = simple_map.write_basis_function(term=[0], mode='standard')
        
        # String is a key: P_{variable}_{O}rder_{n}
        assert string == "P_0_O_1"
        
        # modifier_log contains "variables" dict for subexpression caching
        assert modifier_log == {
            "variables": {
                "P_0_O_1": "np.polynomial.HermiteE([0.0,1.0])(__x__[...,0])"
            }
        }

    def test_univariate_polynomial_derivative_same_var(self, simple_map):
        """Derivative wrt same variable should return derivative string.
        
        For derivatives, the string contains the full expression (not a key),
        and modifier_log["variables"] contains the derivative polynomial.
        
        The derivative of HermiteE([0.0, 1.0]) = x is HermiteE([1.0]) = 1 (constant).
        """
        string, modifier_log = simple_map.write_basis_function(term=[0], mode='derivative', k=0)
        
        # For order-1 polynomial, derivative is a constant (order 0)
        assert string == "np.polynomial.HermiteE([1.0])(__x__[...,0])"
        assert modifier_log == {
            "variables": {
                "P_0_O_1_DER": "np.polynomial.HermiteE([1.0])(__x__[...,0])"
            }
        }

    def test_univariate_polynomial_derivative_different_var(self, simple_map):
        """Derivative wrt different variable should return zeros.
        
        When taking derivative of a polynomial in variable 0 with respect to
        variable 1 (k=1), the result is zero since x_0 doesn't depend on x_1.
        
        Note: modifier_log still contains the original polynomial expression,
        even though the derivative is zero.
        """
        string, modifier_log = simple_map.write_basis_function(term=[0], mode='derivative', k=1)
        
        assert string == "np.zeros(__x__.shape[:-1])"
        assert modifier_log == {
            "variables": {
                "P_0_O_1": "np.polynomial.HermiteE([0.0,1.0])(__x__[...,0])"
            }
        }

    def test_multivariate_polynomial(self, simple_map):
        """Product of polynomials in multiple variables.
        
        Term [0, 1] represents x_0 * x_1, a product of first-order polynomials.
        The string contains keys separated by " * ", and modifier_log["variables"]
        contains the expression for each unique polynomial.
        """
        # Term [0, 1] means product of polynomials in variables 0 and 1
        string, modifier_log = simple_map.write_basis_function(term=[0, 1], mode='standard')
        
        # String returns keys like "P_0_O_1 * P_1_O_1"
        assert string == "P_0_O_1 * P_1_O_1"
        assert modifier_log == {
            "variables": {
                "P_0_O_1": "np.polynomial.HermiteE([0.0,1.0])(__x__[...,0])",
                "P_1_O_1": "np.polynomial.HermiteE([0.0,1.0])(__x__[...,1])"
            }
        }


class TestWriteBasisFunctionSpecialTerms:
    def test_rbf_standard(self, simple_map):
        """RBF (Radial Basis Function) special term should return valid string.
        
        Special terms are specified as strings like "RBF 0" where:
        - "RBF" is the term type (Radial Basis Function)
        - "0" is the variable index
        
        The modifier_log contains {"ST": 0} where:
        - "ST" indicates this is a Special Term
        - The value (0) is the variable index parsed from the term string
        
        """
        string, modifier_log = simple_map.write_basis_function(term="RBF 0", mode='standard')
        
        assert string == "np.exp(-((__x__[...,0] - __mu__)/__scale__)**2/2)/(__scale__*np.sqrt(2*np.pi))"
        assert modifier_log == {"ST": 0}

    def test_irbf_standard(self, simple_map):
        """iRBF special term should return valid string."""
        string, modifier_log = simple_map.write_basis_function(term="iRBF 0", mode='standard')
        
        assert string == "(1 + scipy.special.erf((__x__[...,0] - __mu__)/(np.sqrt(2)*__scale__)))/2"
        assert modifier_log == {"ST": 0}

    def test_let_standard(self, simple_map):
        """LET (left edge term) should return valid string."""
        string, modifier_log = simple_map.write_basis_function(term="LET 0", mode='standard')
        
        assert string == "((__x__[...,0] - __mu__)*(1-scipy.special.erf((__x__[...,0] - __mu__)/(np.sqrt(2)*__scale__))) - __scale__*np.sqrt(2/np.pi)*np.exp(-((__x__[...,0] - __mu__)/(np.sqrt(2)*__scale__))**2))/2"
        assert modifier_log == {"ST": 0}

    def test_ret_standard(self, simple_map):
        """RET (right edge term) should return valid string."""
        string, modifier_log = simple_map.write_basis_function(term="RET 0", mode='standard')
        
        assert string == "((__x__[...,0] - __mu__)*(1+scipy.special.erf((__x__[...,0] - __mu__)/(np.sqrt(2)*__scale__))) + __scale__*np.sqrt(2/np.pi)*np.exp(-((__x__[...,0] - __mu__)/(np.sqrt(2)*__scale__))**2))/2"
        assert modifier_log == {"ST": 0}

    def test_special_term_derivative_same_var(self, simple_map):
        """Derivative of special term wrt same variable."""
        string, modifier_log = simple_map.write_basis_function(term="RBF 0", mode='derivative', k=0)
        
        assert string == "-(__x__[...,0] - __mu__)/(np.sqrt(2*np.pi)*__scale__**3)*np.exp(-((__x__[...,0]-__mu__)/__scale__)**2/2)"
        assert modifier_log == {"ST": 0}

    def test_special_term_derivative_different_var(self, simple_map):
        """Derivative of special term wrt different variable returns zeros."""
        string, modifier_log = simple_map.write_basis_function(term="RBF 0", mode='derivative', k=1)
        
        assert string == "np.zeros(__x__.shape[:-1])"
        assert modifier_log == {"ST": 0}


class TestWriteBasisFunctionValidation:
    """Tests for input validation."""

    def test_invalid_mode_raises_error(self, simple_map):
        """Invalid mode should raise ValueError."""
        with pytest.raises(ValueError, match="Mode must be either"):
            simple_map.write_basis_function(term=[], mode='invalid')

    def test_derivative_without_k_raises_error(self, simple_map):
        """Derivative mode without k should raise ValueError."""
        with pytest.raises(ValueError, match="specify an integer for k"):
            simple_map.write_basis_function(term=[], mode='derivative', k=None)

    def test_derivative_with_non_integer_k_raises_error(self, simple_map):
        """Derivative mode with non-integer k should raise ValueError."""
        with pytest.raises(ValueError, match="specify an integer for k"):
            simple_map.write_basis_function(term=[], mode='derivative', k=1.5)

    def test_unknown_special_term_raises_error(self, simple_map):
        """Unknown special term type should raise ValueError."""
        with pytest.raises(ValueError, match="Special term"):
            simple_map.write_basis_function(term="UNKNOWN 0", mode='standard')
