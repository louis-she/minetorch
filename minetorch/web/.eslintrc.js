module.exports = {
  root: true,
  parserOptions: {
    parser: 'babel-eslint'
  },
  env: {
    browser: true,
    node: true,
    mocha: true
  },
  globals: {
    expect: true,
    JSONEditor: true,
    sceditor: true,
    $: true
  },
  extends: [
    'plugin:vue/recommended',
    'standard'
  ],
  plugins: [
    'vue'
  ],
  rules: {
    'no-new': 0,
    'generator-star-spacing': 'off',
    'no-unused-vars': 'off',
    'no-mixed-operators': 'off',
    'space-before-function-paren': 'off',
    'vue/max-attributes-per-line': 'off',
    'no-debugger': process.env.NODE_ENV === 'production' ? 'error' : 'off',
    'vue/html-self-closing': ['error', {
      'html': {
        'void': 'never',
        'normal': 'never',
        'component': 'always'
      },
      'svg': 'always',
      'math': 'always'
    }]
  }
}
