'use strict'

const webpack = require('webpack');
const path = require('path');

module.exports = {
  mode: 'production',
  entry: {
    vendor: ['vue', 'vue-router', 'babel-polyfill', 'axios']
  },
  output: {
    path: path.resolve('static/js/lib'),
    filename: 'vendor-[chunkhash:7].js?',
    library: '_dll_[name]'
  },
  plugins: [
    new webpack.DllPlugin({path: path.resolve('build/vendor-manifest.json'), name: '_dll_[name]', context: __dirname})
  ]
};
