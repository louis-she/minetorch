<template>
  <div class="m-term">
    <div
      ref="term"
      class="m-term-client"></div>
  </div>
</template>

<script>
import { Terminal } from 'xterm'
import * as attach from 'xterm/lib/addons/attach/attach'
Terminal.applyAddon(attach)

export default {
  props: {
    rows: {
      type: Array,
      default: () => []
    },
    // socket地址
    attach: {
      type: String,
      default: () => ''
    }
  },
  data () {
    return {
      term: null,
      socket: null
    }
  },
  watch: {
    rows (val) {
      val.forEach(v => {
        this.term.writeln(v)
      })
    }
  },
  mounted () {
    this.term = new Terminal({
      theme: {
        background: '#212121'
      }
    })

    if (this.attach) {
      this.socket = new WebSocket(this.attach)
      this.term.attach(this.socket)
    }

    this.term.open(this.$refs.term)
  },
  beforeDestroy () {
    if (this.attach) {
      this.term.detach(this.socket)
    }
  }
}
</script>

<style src="xterm/dist/xterm.css"></style>
<style lang="scss">
.m-term {
  .xterm {
    padding: 8px 10px;
  }
}
</style>
