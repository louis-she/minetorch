'use strict'

const path = require('path')
const utils = require('./utils')

const os = require('os')
const ifaces = os.networkInterfaces()
let localhost = '0.0.0.0'

module.exports = {
  devServer: {
    host: localhost,
    port: 3100,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:5000',
        changeOrigin: true
      }
    }
  },
  assetsRoot: utils.resolve('dist/'),
  assetsJsPath: 'static/js/',
  assetsCssPath: 'static/css/',
  assetsImgPath: 'static/image/',
  assetsPublicPath: process.env.BUILD_ENV === 'production' ? '' : ''
}
