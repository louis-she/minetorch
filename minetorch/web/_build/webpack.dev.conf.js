'use strict'

process.env.NODE_ENV = 'development';

const webpack = require('webpack')
const merge = require('webpack-merge')
const baseConfig = require('./webpack.base.conf')
const config = require('./config')

module.exports = merge(baseConfig, {
  mode: process.env.NODE_ENV,
  devServer: {
    clientLogLevel: 'none',
    hot: true,
    contentBase: config.assetsRoot,
    compress: true,
    host: config.devServer.host,
    port: config.devServer.port,
    open: true,
    overlay: { warnings: false, errors: true },
    watchOptions: {
      poll: true
    },
    proxy: config.devServer.proxy,
    historyApiFallback: true
  },
  plugins: [
    new webpack.HotModuleReplacementPlugin()
  ]
})
